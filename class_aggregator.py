"""
Class Schedule Aggregator untuk agregasi jadwal asisten dan kelas.
Menyediakan:
1. Matriks kesibukan kelas & matriks jadwal kosong (Heatmap Data)
2. Matriks jadwal per asisten (Asisten x Jam)
3. Free Slot Finder (Pencari jam kosong bersama untuk shift jaga/piket/praktikum)
4. Rekapitulasi sebaran mata kuliah & peminatan
5. Deteksi bentrok jadwal
6. Ekspor Excel multi-sheet (.xlsx) dengan nama pendek asisten
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from data_parser import KSMParsedData, Course, StudentInfo, is_course_excluded_from_mapping

logger = logging.getLogger("Class_Aggregator")

DAYS = ["SENIN", "SELASA", "RABU", "KAMIS", "JUMAT", "SABTU"]


def parse_time_float(t_str: str) -> float:
    """Konversi '10:30' atau '10:30:00' ke float (10.5)."""
    parts = t_str.strip().split(":")
    return int(parts[0]) + int(parts[1]) / 60.0


def float_to_time_str(val: float) -> str:
    """Konversi float 10.5 ke '10:30'."""
    h = int(val)
    m = int(round((val - h) * 60))
    return f"{h:02d}:{m:02d}"


def get_short_name(obj) -> str:
    """Ekstraksi nama pendek asisten secara aman dari StudentInfo atau KSMParsedData."""
    if obj is None:
        return "Asisten"
    si = getattr(obj, "student_info", obj)
    if hasattr(si, "short_name") and isinstance(si.short_name, str):
        return si.short_name
    if hasattr(si, "nickname") and si.nickname:
        nick = si.nickname.strip()
        if nick.upper() == "SYIHAM MUHAMMAD RAFI":
            return "Syiham"
        return nick
    if hasattr(si, "name") and si.name:
        parts = str(si.name).strip().split()
        if parts:
            return parts[0].capitalize()
    return "Asisten"


@dataclass
class FreeSlot:
    day: str
    start_time: str
    end_time: str
    duration_hours: float
    free_student_count: int
    total_students: int
    free_percentage: float
    free_student_names: List[str] = field(default_factory=list)
    busy_student_names: List[str] = field(default_factory=list)
    busy_details_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class AssistantConflictInfo:
    course_name: str
    course_code: str
    schedule_time: str
    room: str
    conflict_start: str
    conflict_end: str
    conflict_duration_hours: float
    explanation: str


@dataclass
class AssistantStrictStatus:
    name: str
    full_name: str
    nim: str
    is_completely_free: bool
    conflicts: List[AssistantConflictInfo] = field(default_factory=list)


@dataclass
class StrictDayResult:
    day: str
    target_start: str
    target_end: str
    target_duration: float
    completely_free_assistants: List[str]
    busy_assistants: List[AssistantStrictStatus]
    total_assistants: int

    @property
    def free_percentage(self) -> float:
        if self.total_assistants == 0:
            return 0.0
        return round((len(self.completely_free_assistants) / self.total_assistants) * 100, 1)


@dataclass
class StrictAvailabilityReport:
    target_start_time: str
    target_end_time: str
    target_duration_hours: float
    days_checked: List[str]
    per_day_results: Dict[str, StrictDayResult]
    all_days_free_assistants: List[str]
    total_assistants: int
    matrix_df: pd.DataFrame


class ClassAggregator:
    """
    Agregator jadwal asisten/mahasiswa untuk mengelola jadwal kuliah,
    heatmap ketersediaan, dan pencarian jadwal kosong bersama.
    """

    def __init__(
        self,
        start_hour: float = 7.0,
        end_hour: float = 21.0,
        slot_minutes: int = 30,
        exclude_non_mapping: bool = True
    ):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.slot_step = slot_minutes / 60.0
        self.exclude_non_mapping = exclude_non_mapping
        self.students_data: List[KSMParsedData] = []
        self._init_time_slots()
        self._matrix_details: Dict[str, List[List[str]]] = {}
        self._busy_courses: Dict[str, List[Dict[str, str]]] = {}
        self._free_details: Dict[str, List[List[str]]] = {}

    def get_effective_courses(self, student: KSMParsedData) -> List[Course]:
        """
        Daftar mata kuliah yang diperhitungkan dalam heatmap & pemetaan jadwal asisten.
        Otomatis mengecualikan 'CAPSTONE DESIGN AND PROJECT' dan 'MERDEKA BELAJAR - MAGANG'
        jika exclude_non_mapping bernilai True.
        """
        if not self.exclude_non_mapping:
            return student.courses
        return [c for c in student.courses if not is_course_excluded_from_mapping(c)]

    def _init_time_slots(self):
        self.time_slots = []
        curr = self.start_hour
        while curr < self.end_hour:
            self.time_slots.append(curr)
            curr += self.slot_step

    def add_student(self, parsed_ksm: KSMParsedData):
        """Tambahkan hasil parsing KSM seorang asisten/mahasiswa."""
        if parsed_ksm and parsed_ksm.courses:
            self.students_data.append(parsed_ksm)

    def set_students(self, students: List[KSMParsedData]):
        """Set seluruh data asisten (22-34 mahasiswa)."""
        self.students_data = [s for s in students if s and s.courses]

    @property
    def total_students(self) -> int:
        return len(self.students_data)

    @property
    def all_assistant_names(self) -> List[str]:
        """Daftar seluruh nama pendek asisten."""
        names = []
        for s in self.students_data:
            s_name = get_short_name(s)
            if s_name not in names:
                names.append(s_name)
        return sorted(names)

    def build_occupancy_matrix(self) -> pd.DataFrame:
        """
        Bangun matriks jumlah asisten yang SIBUK (ada kuliah) pada setiap slot waktu per hari.
        Index: Time Slot ('07:00-07:30', dll.)
        Columns: SENIN, SELASA, RABU, KAMIS, JUMAT, SABTU
        """
        matrix = {d: [0] * len(self.time_slots) for d in DAYS}
        matrix_details = {d: [[] for _ in range(len(self.time_slots))] for d in DAYS}
        busy_courses = {d: [{} for _ in range(len(self.time_slots))] for d in DAYS}

        all_names = self.all_assistant_names

        for s in self.students_data:
            si = s.student_info
            s_name = get_short_name(si)
            for c in self.get_effective_courses(s):
                day = c.schedule_day.upper()
                if day not in DAYS or not c.schedule_time or "-" not in c.schedule_time:
                    continue

                try:
                    p1, p2 = c.schedule_time.split("-")
                    start_f = parse_time_float(p1)
                    end_f = parse_time_float(p2)
                except Exception:
                    continue

                for idx, slot_start in enumerate(self.time_slots):
                    slot_end = slot_start + self.slot_step
                    # Jika ada irisan waktu antara jadwal MK dan slot
                    if max(slot_start, start_f) < min(slot_end, end_f):
                        if s_name not in matrix_details[day][idx]:
                            matrix[day][idx] += 1
                            matrix_details[day][idx].append(s_name)
                            course_label = f"{c.name[:20]} ({c.room})" if c.room else c.name[:20]
                            busy_courses[day][idx][s_name] = course_label

        time_labels = [
            f"{float_to_time_str(t)}-{float_to_time_str(t + self.slot_step)}"
            for t in self.time_slots
        ]
        df = pd.DataFrame(matrix, index=time_labels)
        self._matrix_details = matrix_details
        self._busy_courses = busy_courses

        # Bangun juga data asisten yang KOSONG (free)
        free_details = {d: [[] for _ in range(len(self.time_slots))] for d in DAYS}
        for d in DAYS:
            for idx in range(len(self.time_slots)):
                busy_set = set(matrix_details[d][idx])
                free_details[d][idx] = [n for n in all_names if n not in busy_set]
        self._free_details = free_details

        return df

    def build_free_matrix(self) -> pd.DataFrame:
        """
        Bangun matriks jumlah asisten yang KOSONG / FREE (tidak ada kuliah)
        pada setiap slot waktu per hari.
        """
        busy_df = self.build_occupancy_matrix()
        total = self.total_students
        free_df = total - busy_df
        return free_df

    def get_slot_assistant_details(self, day: str, slot_idx: int) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Kembalikan (free_assistants, busy_assistants, busy_courses_map) untuk slot tertentu.
        """
        if not self._matrix_details:
            self.build_occupancy_matrix()

        day_up = day.upper()
        if day_up not in DAYS or slot_idx >= len(self.time_slots):
            return [], [], {}

        busy = self._matrix_details.get(day_up, [])[slot_idx] if slot_idx < len(self._matrix_details.get(day_up, [])) else []
        free = self._free_details.get(day_up, [])[slot_idx] if slot_idx < len(self._free_details.get(day_up, [])) else []
        busy_map = self._busy_courses.get(day_up, [])[slot_idx] if slot_idx < len(self._busy_courses.get(day_up, [])) else {}
        return free, busy, busy_map

    def build_assistant_schedule_matrix(self, day: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Bangun matriks ketersediaan per asisten untuk hari tertentu.
        Returns:
            - text_df: Nilai 'KOSONG' atau keterangan MK (untuk tabel / display)
            - numeric_df: Nilai 0 (KOSONG / Free) atau 1 (SIBUK / Kuliah) untuk heatmap!
        """
        self.build_occupancy_matrix()
        day_up = day.upper()
        time_labels = [
            f"{float_to_time_str(t)}-{float_to_time_str(t + self.slot_step)}"
            for t in self.time_slots
        ]
        all_assistants = self.all_assistant_names

        text_data = {t: ["KOSONG"] * len(all_assistants) for t in time_labels}
        num_data = {t: [0] * len(all_assistants) for t in time_labels}

        name_to_idx = {name: idx for idx, name in enumerate(all_assistants)}

        if day_up in DAYS and day_up in self._matrix_details:
            for s_idx, t_label in enumerate(time_labels):
                busy_list = self._matrix_details[day_up][s_idx]
                courses_dict = self._busy_courses[day_up][s_idx]
                for b_name in busy_list:
                    if b_name in name_to_idx:
                        row_idx = name_to_idx[b_name]
                        c_info = courses_dict.get(b_name, "Kuliah")
                        text_data[t_label][row_idx] = c_info
                        num_data[t_label][row_idx] = 1

        text_df = pd.DataFrame(text_data, index=all_assistants)
        num_df = pd.DataFrame(num_data, index=all_assistants)
        return text_df, num_df

    def find_free_slots(
        self,
        min_duration_hours: float = 1.0,
        max_busy_students: int = 0,
        day_filter: Optional[str] = None
    ) -> List[FreeSlot]:
        """
        Temukan slot waktu kosong bersama untuk seluruh kelas/asisten.
        max_busy_students: toleransi jumlah mahasiswa yang sedang ada kelas (default 0 = semua bebas).
        """
        matrix_df = self.build_occupancy_matrix()
        free_slots: List[FreeSlot] = []
        total = self.total_students

        if total == 0:
            return []

        days_to_check = [day_filter.upper()] if (day_filter and day_filter.upper() in DAYS) else DAYS

        all_names = set(self.all_assistant_names)

        for day in days_to_check:
            busy_counts = matrix_df[day].values
            in_slot = False
            slot_start_idx = 0
            current_busy_students = set()
            current_busy_map = {}

            for i, count in enumerate(busy_counts):
                is_free = count <= max_busy_students
                if is_free:
                    if not in_slot:
                        in_slot = True
                        slot_start_idx = i
                        current_busy_students = set(self._matrix_details[day][i])
                        current_busy_map = dict(self._busy_courses[day][i])
                    else:
                        current_busy_students.update(self._matrix_details[day][i])
                        current_busy_map.update(self._busy_courses[day][i])
                else:
                    if in_slot:
                        # Tutup slot
                        duration = (i - slot_start_idx) * self.slot_step
                        if duration >= min_duration_hours:
                            t_start = float_to_time_str(self.time_slots[slot_start_idx])
                            t_end = float_to_time_str(self.time_slots[i])
                            free_list = sorted(list(all_names - current_busy_students))
                            busy_list = sorted(list(current_busy_students))
                            free_count = len(free_list)
                            free_slots.append(FreeSlot(
                                day=day,
                                start_time=t_start,
                                end_time=t_end,
                                duration_hours=round(duration, 1),
                                free_student_count=free_count,
                                total_students=total,
                                free_percentage=round((free_count / total) * 100, 1),
                                free_student_names=free_list,
                                busy_student_names=busy_list,
                                busy_details_map=current_busy_map
                            ))
                        in_slot = False

            # Jika slot terbuka sampai akhir jam
            if in_slot:
                duration = (len(busy_counts) - slot_start_idx) * self.slot_step
                if duration >= min_duration_hours:
                    t_start = float_to_time_str(self.time_slots[slot_start_idx])
                    t_end = float_to_time_str(self.end_hour)
                    free_list = sorted(list(all_names - current_busy_students))
                    busy_list = sorted(list(current_busy_students))
                    free_count = len(free_list)
                    free_slots.append(FreeSlot(
                        day=day,
                        start_time=t_start,
                        end_time=t_end,
                        duration_hours=round(duration, 1),
                        free_student_count=free_count,
                        total_students=total,
                        free_percentage=round((free_count / total) * 100, 1),
                        free_student_names=free_list,
                        busy_student_names=busy_list,
                        busy_details_map=current_busy_map
                    ))

        # Urutkan berdasarkan persentase bebas tertinggi, lalu durasi terpanjang
        free_slots.sort(key=lambda s: (-s.free_percentage, -s.duration_hours))
        return free_slots

    def find_strictly_free_assistants(
        self,
        start_time: str,
        end_time: str,
        days: Optional[List[str]] = None
    ) -> StrictAvailabilityReport:
        """
        Mencari asisten yang BENAR-BENAR KOSONG (100% free) sepanjang rentang waktu [start_time, end_time].
        Kriteria ketat: jika ada kuliah yang bertabrakan dengan rentang ini (meskipun hanya 15-30 menit,
        atau asisten hanya kosong di 1-2 jam pertama/terakhir), asisten tersebut TIDAK DIANGGAP KOSONG.
        """
        t_start_f = parse_time_float(start_time)
        t_end_f = parse_time_float(end_time)
        duration = round(t_end_f - t_start_f, 2)

        if days is None:
            days = DAYS
        else:
            days = [d.upper() for d in days if d.upper() in DAYS]
            if not days:
                days = DAYS

        total = self.total_students
        all_names = self.all_assistant_names

        per_day_results: Dict[str, StrictDayResult] = {}
        daily_free_sets = []
        matrix_rows = {name: {} for name in all_names}

        for d in days:
            completely_free_names = []
            busy_statuses: List[AssistantStrictStatus] = []

            for s in self.students_data:
                s_name = get_short_name(s)
                si = s.student_info
                conflicts: List[AssistantConflictInfo] = []

                for c in self.get_effective_courses(s):
                    if c.schedule_day.upper() == d and c.schedule_time and "-" in c.schedule_time:
                        try:
                            p1, p2 = c.schedule_time.split("-")
                            c_start_f = parse_time_float(p1)
                            c_end_f = parse_time_float(p2)

                            # Cek irisan waktu: max(t_start, c_start) < min(t_end, c_end)
                            overlap_start = max(t_start_f, c_start_f)
                            overlap_end = min(t_end_f, c_end_f)

                            if overlap_start < overlap_end:
                                overlap_dur = round(overlap_end - overlap_start, 2)

                                # Penjelasan manusiawi mengenai bentrok
                                exp_parts = []
                                if c_start_f > t_start_f:
                                    free_before = round(c_start_f - t_start_f, 1)
                                    exp_parts.append(f"Hanya kosong {free_before} jam di awal (kuliah mulai {float_to_time_str(c_start_f)})")
                                if c_end_f < t_end_f:
                                    free_after = round(t_end_f - c_end_f, 1)
                                    exp_parts.append(f"Hanya kosong {free_after} jam di akhir (kuliah selesai {float_to_time_str(c_end_f)})")
                                if not exp_parts:
                                    exp_parts.append(f"Ada kuliah sepanjang jam shift ({c.schedule_time})")

                                conflicts.append(AssistantConflictInfo(
                                    course_name=c.name,
                                    course_code=c.code,
                                    schedule_time=c.schedule_time,
                                    room=c.room,
                                    conflict_start=float_to_time_str(overlap_start),
                                    conflict_end=float_to_time_str(overlap_end),
                                    conflict_duration_hours=overlap_dur,
                                    explanation="; ".join(exp_parts)
                                ))
                        except Exception:
                            pass

                if not conflicts:
                    completely_free_names.append(s_name)
                    matrix_rows[s_name][d] = "🟢 KOSONG PENUH"
                else:
                    busy_statuses.append(AssistantStrictStatus(
                        name=s_name,
                        full_name=si.name,
                        nim=si.nim,
                        is_completely_free=False,
                        conflicts=conflicts
                    ))
                    first_c = conflicts[0]
                    short_c_name = first_c.course_name[:14]
                    matrix_rows[s_name][d] = f"🔴 {short_c_name} ({first_c.schedule_time})"

            completely_free_names = sorted(completely_free_names)
            busy_statuses.sort(key=lambda x: x.name)
            daily_free_sets.append(set(completely_free_names))

            per_day_results[d] = StrictDayResult(
                day=d,
                target_start=start_time,
                target_end=end_time,
                target_duration=duration,
                completely_free_assistants=completely_free_names,
                busy_assistants=busy_statuses,
                total_assistants=total
            )

        if daily_free_sets:
            all_days_free = sorted(list(set.intersection(*daily_free_sets)))
        else:
            all_days_free = []

        matrix_df = pd.DataFrame.from_dict(matrix_rows, orient="index")
        matrix_df = matrix_df.reindex(columns=days)

        return StrictAvailabilityReport(
            target_start_time=start_time,
            target_end_time=end_time,
            target_duration_hours=duration,
            days_checked=days,
            per_day_results=per_day_results,
            all_days_free_assistants=all_days_free,
            total_assistants=total,
            matrix_df=matrix_df
        )

    def get_course_breakdown(self) -> pd.DataFrame:
        """
        Daftar rekapitulasi mata kuliah di kelas: kode, nama, SKS, jadwal, ruangan,
        jumlah asisten yang mengambil, dan persentase kelas.
        """
        course_map: Dict[str, Dict[str, Any]] = {}

        for s in self.students_data:
            s_name = get_short_name(s)
            for c in s.courses:
                key = f"{c.code}_{c.schedule_day}_{c.schedule_time}"
                if key not in course_map:
                    is_ex = is_course_excluded_from_mapping(c)
                    course_map[key] = {
                        "Kode MK": c.code,
                        "Nama Mata Kuliah": c.name,
                        "SKS": c.sks,
                        "Kelas": c.class_info,
                        "Hari": c.schedule_day,
                        "Waktu": c.schedule_time,
                        "Ruangan": c.room,
                        "Status Mapping": "⚪ Dikecualikan dari Heatmap" if is_ex else "🟢 Aktif di Heatmap",
                        "Peserta": [],
                    }
                course_map[key]["Peserta"].append(s_name)

        rows = []
        for item in course_map.values():
            peserta = sorted(list(set(item["Peserta"])))
            count = len(peserta)
            rows.append({
                "Kode MK": item["Kode MK"],
                "Nama Mata Kuliah": item["Nama Mata Kuliah"],
                "SKS": item["SKS"],
                "Kelas": item["Kelas"],
                "Hari": item["Hari"],
                "Waktu": item["Waktu"],
                "Ruangan": item["Ruangan"],
                "Status Mapping": item["Status Mapping"],
                "Jumlah Mahasiswa": count,
                "Persentase Kelas": f"{(count / max(self.total_students, 1)) * 100:.1f}%",
                "Daftar Asisten": ", ".join(peserta)
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by=["Jumlah Mahasiswa", "Hari"], ascending=[False, True])
        return df

    def detect_student_clashes(self) -> List[Dict[str, Any]]:
        """Deteksi apakah ada asisten yang mengalami bentrok jadwal kuliah."""
        clashes = []
        for s in self.students_data:
            s_name = get_short_name(s)
            courses = s.courses
            for i in range(len(courses)):
                for j in range(i + 1, len(courses)):
                    c1 = courses[i]
                    c2 = courses[j]
                    if c1.schedule_day == c2.schedule_day and c1.schedule_time and c2.schedule_time:
                        try:
                            s1, e1 = c1.schedule_time.split("-")
                            s2, e2 = c2.schedule_time.split("-")
                            f_s1, f_e1 = parse_time_float(s1), parse_time_float(e1)
                            f_s2, f_e2 = parse_time_float(s2), parse_time_float(e2)
                            if max(f_s1, f_s2) < min(f_e1, f_e2):
                                clashes.append({
                                    "Asisten": s_name,
                                    "NIM": s.student_info.nim,
                                    "Hari": c1.schedule_day,
                                    "MK 1": f"{c1.code} - {c1.name} ({c1.schedule_time})",
                                    "MK 2": f"{c2.code} - {c2.name} ({c2.schedule_time})",
                                })
                        except Exception:
                            pass
        return clashes

    def export_to_excel(self, output_path: str = "output/Rekap_Jadwal_Kelas.xlsx") -> str:
        """
        Ekspor seluruh data kelas & asisten ke file Excel multi-sheet terformat rapi:
        - Sheet 1: Daftar Asisten (Nama Pendek & Lengkap)
        - Sheet 2: Matriks Jadwal Kosong (Ketersediaan Asisten)
        - Sheet 3: Matriks Jam Sibuk (Kuliah)
        - Sheet 4: Rekomendasi Jam Kosong (Lengkap dengan Nama Asisten Bebas)
        - Sheet 5: Sebaran Mata Kuliah
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 1. Daftar Asisten
        student_rows = []
        for idx, s in enumerate(self.students_data, 1):
            si = s.student_info
            student_rows.append({
                "No": idx,
                "Nama Panggilan": get_short_name(si),
                "Nama Lengkap": si.name,
                "NIM": si.nim,
                "Program Studi": si.program_study,
                "Kelas": si.class_id or si.cohort,
                "Dosen Wali": si.advisor,
                "Total SKS": si.total_sks or s.total_calculated_sks,
                "Jumlah MK": len(s.courses),
                "Metode OCR": s.extraction_method
            })
        df_students = pd.DataFrame(student_rows)

        # 2. Matriks Jam Kosong & Sibuk
        df_busy_matrix = self.build_occupancy_matrix()
        df_free_matrix = self.build_free_matrix()

        # 3. Rekomendasi Jam Kosong
        free_slots = self.find_free_slots(min_duration_hours=1.0, max_busy_students=2)
        free_rows = [
            {
                "Hari": fs.day,
                "Jam Mulai": fs.start_time,
                "Jam Selesai": fs.end_time,
                "Durasi (Jam)": fs.duration_hours,
                "Asisten Kosong (Jumlah)": f"{fs.free_student_count} / {fs.total_students}",
                "Persentase Kosong": f"{fs.free_percentage}%",
                "Daftar Asisten Bebas / Kosong": ", ".join(fs.free_student_names),
                "Asisten Berhalangan (Ada Kelas)": ", ".join(fs.busy_student_names) if fs.busy_student_names else "Nihil (Semua Bebas)"
            }
            for fs in free_slots
        ]
        df_free = pd.DataFrame(free_rows)

        # 4. Sebaran MK
        df_courses = self.get_course_breakdown()

        # Tulis ke Excel
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_students.to_excel(writer, sheet_name="Daftar Asisten", index=False)
            df_free_matrix.to_excel(writer, sheet_name="Matriks Asisten Kosong")
            df_busy_matrix.to_excel(writer, sheet_name="Matriks Asisten Kuliah")
            df_free.to_excel(writer, sheet_name="Rekomendasi Jam Kosong", index=False)
            df_courses.to_excel(writer, sheet_name="Rekap Mata Kuliah", index=False)

        logger.info(f"Rekap Excel kelas berhasil disimpan di: {output_path}")
        return output_path

    def export_strict_report_to_excel(
        self,
        report: StrictAvailabilityReport,
        output_path: str = "output/Pencarian_Asisten_Kosong_Penuh.xlsx"
    ) -> str:
        """
        Ekspor hasil pencarian asisten yang benar-benar kosong ke Excel terstruktur:
        - Sheet 1: Matriks Mingguan Ketersediaan
        - Sheet 2: Asisten Bebas Semua Hari
        - Sheet 3: Rekap Ketersediaan Harian
        - Sheet 4: Rincian Bentrok Kuliah
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            report.matrix_df.to_excel(writer, sheet_name="Matriks Mingguan")

            all_free_rows = [
                {"No": idx, "Nama Asisten": name, "Status": f"Bebas Penuh Seluruh Hari ({report.target_start_time}-{report.target_end_time})"}
                for idx, name in enumerate(report.all_days_free_assistants, 1)
            ]
            pd.DataFrame(all_free_rows).to_excel(writer, sheet_name="Bebas Semua Hari", index=False)

            daily_rows = []
            conflict_details_rows = []

            for d, day_res in report.per_day_results.items():
                daily_rows.append({
                    "Hari": d,
                    "Rentang Jam": f"{day_res.target_start} - {day_res.target_end}",
                    "Durasi (Jam)": day_res.target_duration,
                    "Jumlah Asisten Kosong Penuh": len(day_res.completely_free_assistants),
                    "Persentase Kosong": f"{day_res.free_percentage}%",
                    "Daftar Asisten Kosong": ", ".join(day_res.completely_free_assistants),
                    "Jumlah Asisten Berhalangan": len(day_res.busy_assistants)
                })

                for b in day_res.busy_assistants:
                    for c in b.conflicts:
                        conflict_details_rows.append({
                            "Hari": d,
                            "Nama Asisten": b.name,
                            "NIM": b.nim,
                            "Mata Kuliah": c.course_name,
                            "Kode MK": c.course_code,
                            "Jadwal Kuliah": c.schedule_time,
                            "Ruangan": c.room,
                            "Jam Bentrok": f"{c.conflict_start}-{c.conflict_end}",
                            "Durasi Bentrok (Jam)": c.conflict_duration_hours,
                            "Keterangan": c.explanation
                        })

            pd.DataFrame(daily_rows).to_excel(writer, sheet_name="Rekap Ketersediaan Harian", index=False)
            if conflict_details_rows:
                pd.DataFrame(conflict_details_rows).to_excel(writer, sheet_name="Rincian Bentrok Kuliah", index=False)

        logger.info(f"Laporan Excel asisten kosong penuh disimpan di: {output_path}")
        return output_path

