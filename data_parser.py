"""
Parser data KSM (Kartu Studi Mahasiswa & Jadwal Mahasiswa) berbasis analisis tata letak spasial
dan pencocokan teks terstruktur.
"""

import os
import re
import csv
import logging
from typing import List, Optional, Union, Dict, Any, Tuple
from dataclasses import dataclass, field

from ocr_engine import OCRResult, DocumentBox

logger = logging.getLogger("Data_Parser")


@dataclass
class Course:
    """Data mata kuliah dari KSM."""
    number: int
    code: str
    name: str
    sks: int
    class_info: str
    schedule_day: str
    schedule_time: str
    room: str
    instructor: Optional[str] = ""
    raw_line: str = ""

    @property
    def schedule_combined(self) -> str:
        return f"{self.schedule_day}, {self.schedule_time} / {self.room}"

    @property
    def start_time(self) -> str:
        return self.schedule_time.split("-")[0].strip() if "-" in self.schedule_time else "00:00"

    @property
    def end_time(self) -> str:
        return self.schedule_time.split("-")[1].strip() if "-" in self.schedule_time else "00:00"


@dataclass
class StudentInfo:
    """Informasi mahasiswa dari KSM."""
    name: str = ""
    nickname: str = ""
    nim: str = ""
    program_study: str = ""
    semester: str = ""
    cohort: str = ""
    class_id: str = ""
    advisor: str = ""
    total_sks: int = 0
    university: str = "Telkom University"

    @property
    def display_name(self) -> str:
        if self.nickname:
            return f"{self.name} ({self.short_name})"
        return self.name

    @property
    def short_name(self) -> str:
        """Nama pendek asisten yang ringkas untuk heatmap dan jadwal."""
        if self.nickname:
            nick = self.nickname.strip()
            if nick.upper() == "SYIHAM MUHAMMAD RAFI":
                return "Syiham"
            return nick
        if self.name:
            clean = re.sub(r"[^A-Za-z\s]", "", self.name).strip()
            parts = clean.split()
            if parts:
                return parts[0].capitalize()
        return "Asisten"


def get_short_name(obj) -> str:
    """Ekstraksi nama pendek asisten secara aman dari StudentInfo atau KSMParsedData."""
    if obj is None:
        return "Asisten"
    si = getattr(obj, "student_info", obj)
    nick = getattr(si, "nickname", None)
    if nick and isinstance(nick, str) and nick.strip():
        clean_nick = nick.strip()
        if clean_nick.upper() == "SYIHAM MUHAMMAD RAFI":
            return "Syiham"
        return clean_nick
    name = getattr(si, "name", None)
    if name and isinstance(name, str) and name.strip():
        parts = re.sub(r"[^A-Za-z\s]", "", name).strip().split()
        if parts:
            return parts[0].capitalize()
    return "Asisten"


def normalize_course_name(course_name: str, code: str = "") -> str:
    """
    Standardisasi nama mata kuliah Telkom University agar bersih dan konsisten
    dari artefak spasi, pemotongan baris, atau noise OCR multi-kolom.
    """
    c_upper = (course_name or "").upper().strip()
    code_upper = (code or "").upper().strip()

    # 1. Capstone Design and Project
    if code_upper == "BZK4AAC4" or "CAPSTONE" in c_upper:
        return "CAPSTONE DESIGN AND PROJECT"

    # 2. Merdeka Belajar - Magang
    if code_upper in ["UHKXAEB5", "UHKXBEB5", "BZKXBEB3", "BZKXAEB3", "BZKXCEB2"] or "MERDEKA BELAJAR" in c_upper:
        return "MERDEKA BELAJAR - MAGANG"
    if "MAGANG" in c_upper and "SERTIFIKASI" not in c_upper and "PELATIHAN" not in c_upper:
        return "MERDEKA BELAJAR - MAGANG"

    # 3. Mata Kuliah Umum Lainnya
    if code_upper == "BBK4BAB3" or ("PELATIHAN" in c_upper and "SERTIFIKASI" in c_upper):
        return "PELATIHAN DAN SERTIFIKASI"
    if code_upper == "BBK4AAB2" or "METODE PENELITIAN" in c_upper:
        return "METODE PENELITIAN DAN PENYUSUNAN KARYA ILMIAH"
    if code_upper == "BBK3MBB3" or "DEEP LEARNING" in c_upper:
        return "PENGANTAR DEEP LEARNING"
    if code_upper == "BBK4GBB3" or "BIG DATA" in c_upper:
        return "PENGELOLAAN BIG DATA"
    if code_upper == "UAKXACB2" or "AGAMA" in c_upper:
        return "PENDIDIKAN AGAMA ISLAM"
    if code_upper == "UCKXBDB2" or "KEWIRAUSAHAAN" in c_upper:
        return "KEWIRAUSAHAAN"

    return course_name


def is_course_excluded_from_mapping(course_or_name: Any, code: str = "") -> bool:
    """
    Memeriksa apakah mata kuliah dikecualikan dari heatmap & pemetaan jadwal asisten.
    Mata kuliah yang dikecualikan sesuai instruksi:
    1. CAPSTONE DESIGN AND PROJECT (dan variasinya, kode BZK4AAC4)
    2. MERDEKA BELAJAR - MAGANG (dan variasinya, kode UHKXAEB5, UHKXBEB5, BZKXBEB3, dll.)
    """
    if hasattr(course_or_name, "name"):
        c_name = str(getattr(course_or_name, "name", "")).upper().strip()
        c_code = str(getattr(course_or_name, "code", "")).upper().strip()
    else:
        c_name = str(course_or_name).upper().strip()
        c_code = code.upper().strip()

    # 1. Capstone Design and Project
    if "CAPSTONE" in c_name or c_code == "BZK4AAC4":
        return True

    # 2. Merdeka Belajar - Magang
    if "MERDEKA BELAJAR" in c_name:
        return True
    if "MAGANG" in c_name and "PELATIHAN" not in c_name and "SERTIFIKASI" not in c_name:
        return True
    if c_code in ["UHKXAEB5", "UHKXBEB5", "BZKXBEB3", "BZKXAEB3", "BZKXCEB2"]:
        return True

    return False


@dataclass
class KSMParsedData:
    """Hasil parsing lengkap dari KSM."""
    student_info: StudentInfo
    courses: List[Course] = field(default_factory=list)
    document_date: Optional[str] = None
    raw_text: str = ""
    extraction_method: str = "unknown"

    @property
    def total_calculated_sks(self) -> int:
        return sum(c.sks for c in self.courses)


class KSMDataParser:
    """
    Parser cerdas KSM yang mendukung format:
    1. Kartu Studi Mahasiswa (Standard iGracias KSM, pageid=705)
    2. Jadwal Mahasiswa (Timetable format, pageid=2901 - seperti Yazid Ahnaffauzi)
    3. Pemetaan nama panggilan otomatis dari nama.csv
    """

    DAY_MAP = {
        "SENIN": "SENIN",
        "SELASA": "SELASA",
        "RABU": "RABU",
        "KAMIS": "KAMIS",
        "JUMAT": "JUMAT",
        "SABTU": "SABTU",
        "MINGGU": "MINGGU",
    }

    HEADER_KEYWORDS = {
        "NO.", "NO", "MATA", "KULIAH", "SKS", "KELAS", "KELAS / KELAS",
        "PEMINATAN", "JADWAL", "UNIVERSITAS", "TELKOM", "KARTU", "STUDI", "MAHASISWA",
        "TOTAL", "TOTAL SKS", "BANDUNG"
    }

    def __init__(self, nickname_csv_path: Optional[str] = "dataset/nama.csv"):
        self.courses: List[Course] = []
        self.student_info = StudentInfo()
        self.nickname_map = self._load_nicknames(nickname_csv_path)

    def _load_nicknames(self, csv_path: Optional[str]) -> Dict[str, str]:
        """Muat pemetaan nama lengkap -> nama panggilan dari nama.csv jika ada."""
        mapping = {}
        if not csv_path or not os.path.exists(csv_path):
            return mapping
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        full_name = row[0].strip().upper()
                        nick = row[1].strip()
                        if full_name and nick:
                            mapping[full_name] = nick
        except Exception as e:
            logger.warning(f"Tidak dapat memuat nama.csv: {e}")
        return mapping

    def _assign_nickname(self, info: StudentInfo):
        """Cocokkan nama mahasiswa dengan database nama panggilan nama.csv."""
        if not info.name:
            return

        clean_name = re.sub(r"\s+", " ", info.name.strip().upper())
        compact_clean = re.sub(r"[^A-Z0-9]", "", clean_name)
        words_clean = set(re.findall(r"\w+", clean_name))

        if not self.nickname_map:
            parts = clean_name.split()
            if parts:
                info.nickname = parts[0].capitalize()
            return

        # 1. Exact match
        if clean_name in self.nickname_map:
            info.nickname = self.nickname_map[clean_name].strip()
            return

        # 2. Compact string match (abaikan spasi, strip, tanda baca)
        # Menangani variasi OCR tanpa spasi seperti 'MUHAMMAD RAKHAALFARUQ' vs 'MUHAMMAD RAKHA ALFARUQ'
        for full_name, nick in self.nickname_map.items():
            compact_full = re.sub(r"[^A-Z0-9]", "", full_name)
            if compact_clean == compact_full:
                info.nickname = nick.strip()
                return

        # 3. Exact Word Token Set match (urutan kata tertukar seperti 'AULIARAHMA PRASETIO YASIFAPUTRI')
        for full_name, nick in self.nickname_map.items():
            words_full = set(re.findall(r"\w+", full_name))
            if words_clean == words_full:
                info.nickname = nick.strip()
                return

        # 4. Token Overlap Jaccard match (overlap >= 2 kata dan rasio >= 50%)
        for full_name, nick in self.nickname_map.items():
            words_full = set(re.findall(r"\w+", full_name))
            overlap = len(words_clean & words_full)
            if overlap >= 2 and (overlap / max(len(words_clean), len(words_full))) >= 0.5:
                info.nickname = nick.strip()
                return

        # 5. Substring match
        for full_name, nick in self.nickname_map.items():
            if full_name in clean_name or clean_name in full_name:
                info.nickname = nick.strip()
                return

        # 6. Fallback jika nama tidak ditemukan di CSV: ambil nama depan
        parts = clean_name.split()
        if parts:
            info.nickname = parts[0].capitalize()

    def parse(self, input_data: Union[OCRResult, str]) -> KSMParsedData:
        """Parse OCRResult atau raw text string."""
        if isinstance(input_data, OCRResult):
            # Cek apakah ini format Jadwal Mahasiswa (pageid=2901 seperti Yazid)
            if "Jadwal Mahasiswa" in input_data.raw_text or "Kode Mata\nKuliah" in input_data.raw_text or "Kode Mata Kuliah" in input_data.raw_text:
                res = self._parse_jadwal_mahasiswa_format(input_data.raw_text)
                if res.courses:
                    self._assign_nickname(res.student_info)
                    return res
            res = self._parse_from_ocr_result(input_data)
        else:
            if "Jadwal Mahasiswa" in input_data or "Kode Mata Kuliah" in input_data:
                res = self._parse_jadwal_mahasiswa_format(input_data)
                if res.courses:
                    self._assign_nickname(res.student_info)
                    return res
            res = self._parse_from_raw_text(input_data)

        self._assign_nickname(res.student_info)
        return res

    def _parse_jadwal_mahasiswa_format(self, text: str) -> KSMParsedData:
        """
        Parser khusus format 'Jadwal Mahasiswa' (pageid=2901) seperti pada dokumen Yazid Ahnaffauzi.
        Struktur: Halaman 1 jadwal grid, Halaman 2 tabel detail MK.
        """
        student_info = StudentInfo()
        # 1. Info Mahasiswa: 102022300273 - MUHAMMAD YAZID AHNAFFAUZI
        m_info = re.search(r"(\d{8,12})\s*-\s*([A-Z\s]+?)(?:\s+Jam|\n|\Z)", text)
        if m_info:
            student_info.nim = m_info.group(1).strip()
            student_info.name = re.sub(r"\s+", " ", m_info.group(2)).strip()

        m_sem = re.search(r"(Semester\s+[^\n]+)", text)
        if m_sem:
            student_info.semester = m_sem.group(1).strip()

        # 2. Ekstrak baris tabel di Halaman 2:
        # No Hari Jam Ruangan KodeMK NamaMK Dosen Kelas
        row_pattern = re.compile(
            r"(\d{1,2})\s*\n\s*(SENIN|SELASA|RABU|KAMIS|JUMAT|SABTU|MINGGU)\s*\n\s*([\d:]+)\s*\n\s*([^\n]+)\s*\n\s*([A-Z0-9]{7,8})\s*\n\s*(.+?)\s*\n\s*([A-Z0-9]{2,4})\s*\n\s*([A-Z0-9\-\n/]+?)(?=\n\d{1,2}\s*\n|\n\s*9/|\Z)",
            re.DOTALL | re.IGNORECASE
        )

        courses: List[Course] = []
        for m in row_pattern.finditer(text):
            num = int(m.group(1))
            day = m.group(2).upper()
            start_time_raw = m.group(3)
            room = m.group(4).strip()
            code = m.group(5).upper()
            name_mk = re.sub(r"\s+", " ", m.group(6)).strip()
            dosen = m.group(7).strip()
            kelas = re.sub(r"\s+", " ", m.group(8)).strip()

            sks = int(code[-1]) if code[-1].isdigit() else 3

            # Hitung waktu selesai berdasarkan jam mulai + SKS jam
            parts = start_time_raw.split(":")
            start_h = int(parts[0])
            start_m = int(parts[1]) if len(parts) > 1 else 0
            end_h = start_h + sks
            time_str = f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{start_m:02d}"

            clean_name = normalize_course_name(name_mk, code)
            course = Course(
                number=num,
                code=code,
                name=clean_name,
                sks=sks,
                class_info=kelas,
                schedule_day=day,
                schedule_time=time_str,
                room=room,
                instructor=dosen,
                raw_line=f"{num} {code} - {clean_name} {sks} {kelas} / {day}, {time_str} / {room}"
            )
            courses.append(course)

        student_info.total_sks = sum(c.sks for c in courses)
        return KSMParsedData(
            student_info=student_info,
            courses=courses,
            document_date=self._extract_date(text),
            raw_text=text,
            extraction_method="jadwal_mahasiswa_table"
        )

    def _cluster_boxes_into_lines(self, boxes: List[DocumentBox], y_threshold: float = 12.0) -> List[dict]:
        """Kelompokkan kotak teks menjadi baris horizontal berdasarkan koordinat Y."""
        if not boxes:
            return []

        sorted_boxes = sorted(boxes, key=lambda b: b.cy)
        lines = []
        curr_line = []
        curr_cy = None

        for b in sorted_boxes:
            if curr_cy is None or abs(b.cy - curr_cy) < y_threshold:
                curr_line.append(b)
                curr_cy = sum(x.cy for x in curr_line) / len(curr_line)
            else:
                curr_line.sort(key=lambda x: x.x0)
                lines.append({
                    "cy": curr_cy,
                    "text": " ".join(x.text for x in curr_line),
                    "boxes": curr_line
                })
                curr_line = [b]
                curr_cy = b.cy

        if curr_line:
            curr_line.sort(key=lambda x: x.x0)
            lines.append({
                "cy": curr_cy,
                "text": " ".join(x.text for x in curr_line),
                "boxes": curr_line
            })

        return lines

    def _parse_from_ocr_result(self, ocr_result: OCRResult) -> KSMParsedData:
        """Parse menggunakan informasi spasial DocumentBox."""
        boxes = ocr_result.boxes
        method = ocr_result.extraction_method

        if not boxes:
            return self._parse_from_raw_text(ocr_result.raw_text)

        min_page_x = min(b.x0 for b in boxes)
        max_page_x = max(b.x1 for b in boxes)
        page_w = max(max_page_x - min_page_x, 1.0)

        y_thresh = 10.0 if "words" in method else 20.0
        lines = self._cluster_boxes_into_lines(boxes, y_threshold=y_thresh)
        full_text = " ".join(l["text"] for l in lines)

        # 1. Ekstrak informasi mahasiswa
        student_info = self._extract_student_info_text(full_text)

        # 2. Cari semua baris jadwal
        sched_regex = re.compile(
            r"(SENIN|SELASA|RABU|KAMIS|JUMAT|SABTU|MINGGU)[,\s]+([\d:]{5,8})\s*-\s*([\d:]{5,8})\s*/\s*(.+)",
            re.IGNORECASE
        )

        schedule_entries = []
        for l in lines:
            m = sched_regex.search(l["text"])
            if m:
                schedule_entries.append({
                    "line": l,
                    "day": m.group(1).upper(),
                    "start_time": m.group(2).strip(),
                    "end_time": m.group(3).strip(),
                    "room": m.group(4).strip()
                })

        # 3. Cari semua Course Code candidates di kolom MK (rel_x <= 0.25)
        code_regex = re.compile(r"\b([A-Z]{3}\d[A-Z0-9]{3}\d|[A-Z0-9]{8})\b")
        code_entries = []
        for b in sorted(boxes, key=lambda x: x.cy):
            rel_x = (b.x0 - min_page_x) / page_w
            if rel_x <= 0.25:
                for c in code_regex.findall(b.text):
                    if any(ch.isdigit() for ch in c) and any(ch.isalpha() for ch in c):
                        if c not in ("UNIVERSITAS", "SEMESTER", "PROGRAM", "STUDI"):
                            code_entries.append((c, b.cy, b))

        # Hapus duplikat yang sangat dekat secara Y
        filtered_codes = []
        for c, cy, b in code_entries:
            if not any(abs(cy - prev_cy) < 30 and c == prev_c for prev_c, prev_cy, _ in filtered_codes):
                filtered_codes.append((c, cy, b))

        courses: List[Course] = []
        num_courses = len(schedule_entries)

        for idx, s in enumerate(schedule_entries):
            # Tentukan batas Y: setiap mata kuliah dimulai tepat sebelum posisi kode MK-nya
            if len(filtered_codes) == num_courses:
                curr_code_y = filtered_codes[idx][1]
                y_top = curr_code_y - 25.0
                if idx + 1 < num_courses:
                    y_bottom = filtered_codes[idx + 1][1] - 25.0
                else:
                    y_bottom = curr_code_y + 160.0
                code = filtered_codes[idx][0]
            else:
                prev_cy = schedule_entries[idx - 1]["line"]["cy"] if idx > 0 else 0
                next_cy = schedule_entries[idx + 1]["line"]["cy"] if idx + 1 < len(schedule_entries) else 999999
                y_top = (prev_cy + s["line"]["cy"]) / 2.0 if idx > 0 else s["line"]["cy"] - 80
                y_bottom = (s["line"]["cy"] + next_cy) / 2.0 if idx + 1 < len(schedule_entries) else s["line"]["cy"] + 80
                code = f"MK{idx+1:02d}"

            # Ambil kotak teks dalam range baris ini
            row_boxes = [b for b in boxes if y_top <= b.cy < y_bottom]

            # Pisahkan berdasarkan koordinat relatif halaman (X)
            # Kolom MK: rel_x < 0.34
            # Kolom Kelas: 0.34 s.d. 0.55
            # Kolom Jadwal: rel_x >= 0.55
            mk_tokens = []
            class_tokens = []

            for b in sorted(row_boxes, key=lambda x: (x.cy, x.x0)):
                rel_x = (b.x0 - min_page_x) / page_w
                if rel_x >= 0.55 or sched_regex.search(b.text):
                    continue  # Jadwal
                elif 0.34 <= rel_x < 0.55:
                    class_tokens.append(b.text)
                elif rel_x < 0.34:
                    clean = b.text.strip()
                    if clean.upper() not in self.HEADER_KEYWORDS:
                        mk_tokens.append(clean)

            # Ekstrak Kode jika belum didapat
            if code.startswith("MK"):
                row_text = " ".join(b.text for b in row_boxes)
                c_matches = [
                    c for c in code_regex.findall(row_text)
                    if any(ch.isdigit() for ch in c) and any(ch.isalpha() for ch in c)
                    and c not in ("SENIN", "SELASA", "RABU", "KAMIS", "JUMAT", "SABTU", "MINGGU")
                ]
                if c_matches:
                    code = c_matches[0]

            CLASS_PAT = re.compile(r"BS1[A-Z0-9\-]+|SI-\d+-\S*|\d{2}-EDM|GAB\d+|REG-\S*|01-INT|01-EDM|02-EDM", re.IGNORECASE)

            # Ekstrak Nama MK
            name_parts = []
            for t in mk_tokens:
                t_clean = t.replace(code, "").replace("-", " ").strip()
                t_clean = CLASS_PAT.sub("", t_clean).strip()
                words = [w for w in t_clean.split() if w.upper() not in self.HEADER_KEYWORDS]
                t_clean = " ".join(words).strip()
                t_clean = re.sub(r"^\d{1,2}\s*", "", t_clean).strip()
                if t_clean:
                    name_parts.append(t_clean)

            course_name = " ".join(name_parts).strip()
            course_name = re.sub(r"\s+", " ", course_name).strip()
            if not course_name:
                course_name = f"Mata Kuliah {code}"

            # SKS: pada Telkom Univ, digit terakhir kode MK menentukan SKS
            sks = 3
            if code and code[-1].isdigit():
                val = int(code[-1])
                if 1 <= val <= 6:
                    sks = val

            class_info = " ".join(class_tokens).strip()
            if not class_info:
                class_info = student_info.class_id or "REG"

            time_str = f"{s_time(s['start_time'])}-{s_time(s['end_time'])}"

            norm_course_name = normalize_course_name(course_name, code)
            course = Course(
                number=idx + 1,
                code=code,
                name=norm_course_name,
                sks=sks,
                class_info=class_info,
                schedule_day=s["day"],
                schedule_time=time_str,
                room=s["room"],
                raw_line=f"{idx+1} {code} - {norm_course_name} {sks} {class_info} / {s['day']}, {time_str} / {s['room']}"
            )
            courses.append(course)

        # Fallback jika parsing spasial menghasilkan 0 courses: gunakan regex teks
        if not courses:
            return self._parse_from_raw_text(full_text)

        if student_info.total_sks == 0:
            student_info.total_sks = sum(c.sks for c in courses)

        return KSMParsedData(
            student_info=student_info,
            courses=courses,
            document_date=self._extract_date(full_text),
            raw_text=full_text,
            extraction_method=method
        )

    def _parse_from_raw_text(self, text: str) -> KSMParsedData:
        """Fallback parser berbasis pencocokan pola baris."""
        student_info = self._extract_student_info_text(text)
        courses: List[Course] = []

        norm = re.sub(r"\s+", " ", text.strip())
        pattern = re.compile(
            r"(\d{1,2})\s+([A-Z0-9]{7,8})\s*-?\s*(.*?)\s+(\d{1,2})\s+(\S+)\s*/\s*(SENIN|SELASA|RABU|KAMIS|JUMAT|SABTU|MINGGU)[,\s]+([\d:]+)-([\d:]+)\s*/\s*([^\s]+)",
            re.IGNORECASE
        )

        matches = list(pattern.finditer(norm))
        for m in matches:
            code = m.group(2).upper()
            raw_cname = m.group(3).strip()
            norm_cname = normalize_course_name(raw_cname, code)
            courses.append(Course(
                number=int(m.group(1)),
                code=code,
                name=norm_cname,
                sks=int(m.group(4)),
                class_info=m.group(5).strip(),
                schedule_day=m.group(6).upper(),
                schedule_time=f"{s_time(m.group(7))}-{s_time(m.group(8))}",
                room=m.group(9).strip(),
                raw_line=m.group(0)
            ))

        if student_info.total_sks == 0:
            student_info.total_sks = sum(c.sks for c in courses)

        return KSMParsedData(
            student_info=student_info,
            courses=courses,
            document_date=self._extract_date(text),
            raw_text=text,
            extraction_method="regex_text"
        )

    def _extract_student_info_text(self, text: str) -> StudentInfo:
        """Ekstraksi metadata mahasiswa dari teks dokumen."""
        info = StudentInfo()

        header_text = text
        for split_word in ["No. Mata Kuliah", "Mata Kuliah SKS", "No.\nMata Kuliah", "Jadwal Mahasiswa"]:
            if split_word in text:
                header_text = text.split(split_word)[0]
                break

        # 1. Nama Mahasiswa
        m_name = re.search(r"Nama\s*:\s*(.+?)(?:\s*(?:NIM|Semester))", header_text, re.IGNORECASE | re.DOTALL)
        if m_name:
            info.name = clean_metadata(m_name.group(1)).lstrip(": -").strip()
        elif "Jadwal Mahasiswa" in text:
            m_alt = re.search(r"(\d{8,12})\s*-\s*([A-Z\s]+?)(?:\s+Jam|\n|\Z)", text)
            if m_alt:
                info.nim = m_alt.group(1).strip()
                info.name = re.sub(r"\s+", " ", m_alt.group(2)).strip()

        # 2. NIM
        if not info.nim:
            m_nim = re.search(r"NIM\s*[:\s]*(\d{8,12})", header_text, re.IGNORECASE)
            if m_nim:
                info.nim = m_nim.group(1).strip()
            else:
                m_nim_fallback = re.search(r"\b(102\d{7,9})\b", header_text)
                if m_nim_fallback:
                    info.nim = m_nim_fallback.group(1).strip()

        # 3. Semester
        m_sem = re.search(r"Semester\s*:\s*([^:\n]+?)(?:\s*(?:Angkatan|Program|Dosen))", header_text, re.IGNORECASE | re.DOTALL)
        if m_sem:
            info.semester = clean_metadata(m_sem.group(1)).lstrip(": -").strip()
        else:
            m_sem_alt = re.search(r"(Semester\s+[^\n]+|Ganjil\s*\d{4}\s*/\s*\d{4}|Genap\s*\d{4}\s*/\s*\d{4})", header_text, re.IGNORECASE)
            if m_sem_alt:
                info.semester = m_sem_alt.group(1).strip()

        # 4. Program Studi
        m_prodi = re.search(r"Program Studi\s*:\s*([^:\n]+?)(?:\s*(?:Semester|Angkatan|Kelas|Dosen))", header_text, re.IGNORECASE | re.DOTALL)
        if m_prodi:
            info.program_study = clean_metadata(m_prodi.group(1)).lstrip(": -").strip()

        # 5. Angkatan / Kelas
        m_ang = re.search(
            r"Angkatan\s*/\s*Kelas(?:\s*/\s*Kelas Peminatan)?\s*:\s*([^:\n]+?)(?:\s*Dosen)",
            header_text, re.IGNORECASE | re.DOTALL
        )
        if m_ang:
            parts = [p.strip() for p in m_ang.group(1).lstrip(": -").split("/") if p.strip()]
            if len(parts) >= 1:
                info.cohort = parts[0]
            if len(parts) >= 2:
                info.class_id = parts[1]

        # 6. Dosen Wali
        m_dos = re.search(r"Dosen Wali\s*:\s*([^:\n]+?)(?:\s*(?:No\.|Kelas|Mata Kuliah|Bukti|\d{1,2}\s+[A-Z0-9]))", header_text, re.IGNORECASE | re.DOTALL)
        if m_dos:
            info.advisor = clean_metadata(m_dos.group(1)).lstrip(": -").strip()

        # 7. Total SKS
        m_sks = re.search(r"Total SKS\s*[:\s]*(\d+)", text, re.IGNORECASE)
        if m_sks:
            info.total_sks = int(m_sks.group(1))

        return info

    def _extract_date(self, text: str) -> Optional[str]:
        """Ekstrak tanggal dokumen."""
        date_patterns = [
            r"Bandung,\s*(\d{2}-\d{2}-\d{4})",
            r"(\d{2}-\d{2}-\d{4})",
            r"(\d{4}-\d{2}-\d{2})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None


def s_time(t: str) -> str:
    """Format waktu '10:30:00' atau '10:30'."""
    t = t.strip()
    parts = t.split(":")
    if len(parts) >= 2:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
    return t


def clean_metadata(text: str) -> str:
    """Bersihkan metadata string."""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text