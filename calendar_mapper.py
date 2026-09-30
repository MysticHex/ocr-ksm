"""
Calendar Mapper untuk visualisasi jadwal perkuliahan perorangan dan kelas (Heatmap),
serta ekspor ke format iCalendar (.ics) standar Google Calendar.
"""

import os
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, date, time, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import numpy as np
import pandas as pd

from icalendar import Calendar, Event

from data_parser import Course

logger = logging.getLogger("Calendar_Mapper")

DAYS_ORDER = ["SENIN", "SELASA", "RABU", "KAMIS", "JUMAT", "SABTU"]
DAY_IND_TO_WEEKDAY = {
    "SENIN": 0, "SELASA": 1, "RABU": 2,
    "KAMIS": 3, "JUMAT": 4, "SABTU": 5, "MINGGU": 6
}

COLOR_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#17B890", "#482677", "#2D708E", "#55C667", "#E76F51",
    "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#6D597A"
]


class CalendarMapper:
    """
    Pembuat kalender visual mingguan dan agregasi kelas.
    """

    def __init__(
        self,
        start_hour: float = 7.0,
        end_hour: float = 21.0,
        output_dir: str = "output"
    ):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _parse_time_range(self, time_range: str) -> Tuple[float, float]:
        parts = time_range.split("-")
        def to_float(s):
            h, m = s.strip().split(":")[:2]
            return int(h) + int(m) / 60.0
        return to_float(parts[0]), to_float(parts[1])

    def create_calendar_image(
        self,
        courses: List[Course],
        student_name: str = "",
        nim: str = "",
        semester: str = "",
        filename: str = "calendar.png"
    ) -> str:
        """Buat gambar jadwal mingguan mahasiswa (PNG) bergaya modern."""
        fig, ax = plt.subplots(figsize=(16, 11), dpi=150)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FFFFFF")

        ax.set_xlim(-0.5, len(DAYS_ORDER) - 0.5)
        ax.set_ylim(self.end_hour, self.start_hour)

        # Header Title
        title = "JADWAL PERKULIAHAN MAHASISWA"
        sub = f"{student_name.upper()} | NIM: {nim} | {semester}" if student_name else semester
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97, color="#1A202C")
        ax.set_title(sub, fontsize=12, color="#4A5568", pad=15)

        # Grid lines
        for x in range(len(DAYS_ORDER)):
            ax.axvline(x=x - 0.5, color="#E2E8F0", linestyle="--", linewidth=1.0)
        ax.axvline(x=len(DAYS_ORDER) - 0.5, color="#E2E8F0", linestyle="--", linewidth=1.0)

        for hour in range(int(self.start_hour), int(self.end_hour) + 1):
            ax.axhline(y=hour, color="#EDF2F7", linewidth=0.8)

        # Draw Courses
        color_idx = 0
        for c in courses:
            day = c.schedule_day.upper()
            if day not in DAYS_ORDER or not c.schedule_time or "-" not in c.schedule_time:
                continue

            try:
                start_h, end_h = self._parse_time_range(c.schedule_time)
            except Exception:
                continue

            day_idx = DAYS_ORDER.index(day)
            color = COLOR_PALETTE[color_idx % len(COLOR_PALETTE)]
            color_idx += 1

            # Rounded block
            block = FancyBboxPatch(
                (day_idx - 0.45, start_h + 0.05),
                0.9, (end_h - start_h) - 0.1,
                boxstyle="round,pad=0.03",
                facecolor=color,
                edgecolor="none",
                alpha=0.92
            )
            ax.add_patch(block)

            center_y = (start_h + end_h) / 2.0
            height = end_h - start_h

            # Text inside block
            text_str = f"{c.code}\n{c.name[:25]}\n{c.schedule_time}\n({c.room})"
            font_size = 8 if height >= 2.0 else 7
            ax.text(
                day_idx, center_y, text_str,
                ha="center", va="center",
                fontsize=font_size, fontweight="bold",
                color="white", wrap=True
            )

        # Ticks
        ax.set_xticks(range(len(DAYS_ORDER)))
        ax.set_xticklabels(DAYS_ORDER, fontsize=11, fontweight="bold", color="#2D3748")

        hours_range = range(int(self.start_hour), int(self.end_hour) + 1)
        ax.set_yticks(hours_range)
        ax.set_yticklabels([f"{h:02d}:00" for h in hours_range], fontsize=9, color="#718096")

        ax.set_ylabel("Waktu", fontsize=11, fontweight="bold", color="#2D3748")

        total_sks = sum(c.sks for c in courses)
        ax.text(
            0.01, 0.02, f"Total SKS: {total_sks} | Total Mata Kuliah: {len(courses)}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF8FF", edgecolor="#BEE3F8")
        )

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Kalender PNG disimpan di: {out_path}")
        return out_path

    def create_class_heatmap_image(
        self,
        matrix_df: pd.DataFrame,
        details_dict: Optional[Dict[str, List[List[str]]]] = None,
        mode: str = "free",  # "free" (jadwal kosong) atau "busy" (jadwal kuliah)
        class_name: str = "Jadwal Asisten",
        total_students: int = 0,
        filename: str = "class_heatmap.png"
    ) -> str:
        """
        Buat visualisasi heatmap matriks jam dengan NAMA PENDEK asisten tertera di dalam sel.
        mode="free": Warna gradasi hijau, menampilkan asisten yang KOSONG (bisa jaga lab/piket).
        mode="busy": Warna gradasi kuning-merah, menampilkan asisten yang sedang KULIAH.
        """
        fig, ax = plt.subplots(figsize=(16, 12), dpi=160)

        cmap = "YlGn" if mode == "free" else "YlOrRd"
        cbar_label = "Jumlah Asisten Kosong (Free)" if mode == "free" else "Jumlah Asisten Ada Kuliah"

        # Bangun annotation array dengan nama pendek asisten
        annot_matrix = np.empty(matrix_df.shape, dtype=object)
        days = list(matrix_df.columns)
        slots = list(matrix_df.index)

        for c_idx, day in enumerate(days):
            day_details = details_dict.get(day, []) if details_dict else []
            for r_idx in range(len(slots)):
                val = int(matrix_df.iloc[r_idx, c_idx])
                names = day_details[r_idx] if r_idx < len(day_details) else []

                if mode == "free":
                    if val == 0:
                        annot_matrix[r_idx, c_idx] = "0 Kosong\n(Semua Sibuk)"
                    elif val == total_students and total_students > 0:
                        annot_matrix[r_idx, c_idx] = f"{val} Kosong\n(Semua Bebas!)"
                    else:
                        preview = ", ".join(names[:2])
                        if len(names) > 2:
                            preview += f"\n+{len(names)-2} lainnya"
                        annot_matrix[r_idx, c_idx] = f"{val} Kosong\n({preview})"
                else:  # mode == "busy"
                    if val == 0:
                        annot_matrix[r_idx, c_idx] = "0 Kuliah\n(Bebas)"
                    else:
                        preview = ", ".join(names[:2])
                        if len(names) > 2:
                            preview += f"\n+{len(names)-2} lainnya"
                        annot_matrix[r_idx, c_idx] = f"{val} Kuliah\n({preview})"

        sns.heatmap(
            matrix_df,
            annot=annot_matrix,
            fmt="",
            cmap=cmap,
            linewidths=0.8,
            linecolor="#CBD5E1",
            cbar_kws={"label": cbar_label},
            annot_kws={"fontsize": 7.5, "weight": "bold"},
            ax=ax
        )

        mode_title = "KETERSEDIAAN / JADWAL KOSONG ASISTEN" if mode == "free" else "KESIBUKAN KULIAH ASISTEN"
        title = f"HEATMAP {mode_title} ({class_name.upper()})"
        sub = f"Total Asisten: {total_students} | Sel memuat jumlah & nama panggilan asisten"
        ax.set_title(f"{title}\n{sub}", fontsize=14, fontweight="bold", pad=16, color="#0F172A")
        ax.set_xlabel("Hari Perkuliahan", fontsize=11, fontweight="bold", labelpad=10, color="#1E293B")
        ax.set_ylabel("Slot Waktu", fontsize=11, fontweight="bold", color="#1E293B")

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Heatmap kelas disimpan di: {out_path}")
        return out_path

    def create_assistant_matrix_heatmap_image(
        self,
        num_df: pd.DataFrame,
        text_df: pd.DataFrame,
        day_name: str = "SENIN",
        filename: str = "assistant_matrix_heatmap.png"
    ) -> str:
        """
        Heatmap matriks individu: Sumbu Y = Nama Pendek Seluruh Asisten, Sumbu X = Waktu.
        Sel Hijau (0) = Kosong (Free), Sel Merah (1) = Kuliah (Sibuk).
        """
        num_assistants = len(num_df.index)
        num_slots = len(num_df.columns)
        fig_height = max(8, num_assistants * 0.45)
        fig_width = max(14, num_slots * 0.6)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)

        # Custom colormap: 0 = Emerald Green (#DCFCE7), 1 = Soft Coral Red (#FEE2E2)
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(["#DCFCE7", "#FEE2E2"])

        annot_matrix = np.empty(num_df.shape, dtype=object)
        for r in range(num_assistants):
            for c in range(num_slots):
                val = num_df.iloc[r, c]
                txt = text_df.iloc[r, c]
                if val == 0:
                    annot_matrix[r, c] = "✓ Kosong"
                else:
                    annot_matrix[r, c] = txt[:14] if txt != "KOSONG" else "Kuliah"

        sns.heatmap(
            num_df,
            annot=annot_matrix,
            fmt="",
            cmap=custom_cmap,
            linewidths=0.6,
            linecolor="#E2E8F0",
            cbar=False,
            annot_kws={"fontsize": 7, "weight": "bold", "color": "#1E293B"},
            ax=ax
        )

        title = f"MATRIKS KETERSEDIAAN ASISTEN — HARI {day_name.upper()}"
        sub = "Hijau (✓ Kosong = Siap Jaga / Piket) | Merah (Kuliah / Ada Kelas)"
        ax.set_title(f"{title}\n{sub}", fontsize=13, fontweight="bold", pad=14, color="#0F172A")
        ax.set_xlabel("Slot Waktu", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel("Nama Asisten (Nama Pendek)", fontsize=10, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8.5, fontweight="bold")

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Heatmap matriks asisten disimpan di: {out_path}")
        return out_path

    def export_to_ics(
        self,
        courses: List[Course],
        student_name: str = "",
        semester_start_date: Optional[date] = None,
        filename: str = "jadwal.ics"
    ) -> str:
        """
        Ekspor daftar mata kuliah ke file iCalendar (.ics) standar RFC 5545
        yang dapat diimpor langsung ke Google Calendar atau Apple Calendar.
        """
        cal = Calendar()
        cal.add("prodid", "-//KSM OCR Schedule Generator//id//")
        cal.add("version", "2.0")

        # Asumsi semester dimulai Senin terdekat jika tidak dispesifikasikan
        if not semester_start_date:
            today = date.today()
            semester_start_date = today - timedelta(days=today.weekday())

        for c in courses:
            day_str = c.schedule_day.upper()
            if day_str not in DAY_IND_TO_WEEKDAY or not c.schedule_time or "-" not in c.schedule_time:
                continue

            target_weekday = DAY_IND_TO_WEEKDAY[day_str]
            days_ahead = (target_weekday - semester_start_date.weekday()) % 7
            first_event_date = semester_start_date + timedelta(days=days_ahead)

            try:
                p1, p2 = c.schedule_time.split("-")
                h1, m1 = [int(x) for x in p1.strip().split(":")[:2]]
                h2, m2 = [int(x) for x in p2.strip().split(":")[:2]]
            except Exception:
                continue

            dt_start = datetime.combine(first_event_date, time(h1, m1))
            dt_end = datetime.combine(first_event_date, time(h2, m2))

            event = Event()
            event.add("summary", f"{c.code} - {c.name}")
            event.add("description", f"Mata Kuliah: {c.name}\nSKS: {c.sks}\nKelas: {c.class_info}\nMahasiswa: {student_name}")
            event.add("location", c.room)
            event.add("dtstart", dt_start)
            event.add("dtend", dt_end)
            # Berulang setiap minggu selama 16 minggu (1 semester)
            event.add("rrule", {"freq": "weekly", "count": 16})

            cal.add_component(event)

        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, "wb") as f:
            f.write(cal.to_ical())

        logger.info(f"File ICS kalender berhasil disimpan di: {out_path}")
        return out_path