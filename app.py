"""
Web Application Interaktif: KSM OCR & Assistant Schedule Manager
Dibangun dengan Streamlit untuk memetakan batch 22-34 KSM Telkom University.
Dilengkapi:
- OCR Cerdas (RapidOCR ONNX & PyMuPDF)
- Pemetaan nama panggilan otomatis dari dataset/nama.csv
- Heatmap Mingguan berisikan nama pendek asisten (Mode Jadwal Kosong & Jadwal Kuliah)
- Matriks Ketersediaan per Asisten (Nama Asisten x Jam)
- Smart Free Schedule Finder (Pencari Jam Kosong Bersama untuk jaga praktikum / piket lab)
"""

import os
import io
import zipfile
import tempfile
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import pandas as pd
import streamlit as st

from ocr_engine import OCREngine
from data_parser import KSMDataParser, KSMParsedData, StudentInfo, is_course_excluded_from_mapping
from class_aggregator import ClassAggregator, parse_time_float, float_to_time_str
from calendar_mapper import CalendarMapper

logging.basicConfig(level=logging.INFO)


def get_short_name(obj) -> str:
    """Ekstraksi nama pendek asisten secara aman dari StudentInfo atau KSMParsedData tanpa rekursi."""
    if obj is None:
        return "Asisten"
    si = getattr(obj, "student_info", obj)

    # 1. Prioritas Nama Panggilan (dari dataset/nama.csv)
    nick = getattr(si, "nickname", None)
    if nick and isinstance(nick, str) and nick.strip():
        clean_nick = nick.strip()
        if clean_nick.upper() == "SYIHAM MUHAMMAD RAFI":
            return "Syiham"
        return clean_nick

    # 2. Fallback ke nama lengkap (ambil kata pertama)
    name = getattr(si, "name", None)
    if name and isinstance(name, str) and name.strip():
        parts = re.sub(r"[^A-Za-z\s]", "", name).strip().split()
        if parts:
            return parts[0].capitalize()

    return "Asisten"


# Kaitkan properti ke class StudentInfo dengan aman
try:
    StudentInfo.short_name = property(lambda self: get_short_name(self))
except Exception:
    pass

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="KSM OCR & Assistant Schedule Manager",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING (Aesthetic UI)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 45%, #0284c7 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.25);
    }
    
    .main-hero h1 {
        color: white !important;
        font-weight: 800;
        font-size: 2.1rem;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .main-hero p {
        color: #e0f2fe;
        font-size: 1.02rem;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
    
    .student-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        margin-right: 6px;
    }

    .nickname-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 700;
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .free-slot-card {
        background: #ffffff;
        border-left: 5px solid #10b981;
        border: 1px solid #e2e8f0;
        border-left-width: 5px;
        border-left-color: #10b981;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }

    .assistant-pill-free {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
        margin: 2px 4px 2px 0;
    }

    .assistant-pill-busy {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        background: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
        margin: 2px 4px 2px 0;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


def render_full_width_image(image_path: str, caption: Optional[str] = None):
    """
    Menampilkan gambar dengan lebar penuh kontainer secara aman dan kompatibel
    lintas versi Streamlit (mencegah TypeError pada use_column_width vs width='stretch').
    """
    # 1. Coba standar terbaru Streamlit 1.5x - 1.6x+
    try:
        st.image(image_path, caption=caption, width="stretch")
        return
    except TypeError:
        pass

    # 2. Coba use_container_width (Streamlit 1.35 - 1.5x)
    try:
        st.image(image_path, caption=caption, use_container_width=True)
        return
    except TypeError:
        pass

    # 3. Coba use_column_width (Streamlit lawas <= 1.30)
    try:
        st.image(image_path, caption=caption, use_column_width=True)
        return
    except TypeError:
        pass

    # 4. Fallback default
    st.image(image_path, caption=caption)


# ==========================================
# INITIALIZE STATE
# ==========================================
if "students_data" not in st.session_state:
    st.session_state["students_data"] = []
if "processed_names" not in st.session_state:
    st.session_state["processed_names"] = []


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/03/Logo_Telkom_University_potrait.png", width=110)
    st.title("⚙️ Kontrol Jadwal Asisten")
    st.markdown("Aplikasi OCR untuk memproses **KSM Asisten** secara batch dan menghasilkan pemetaan jadwal terpadu dengan **nama pendek** dari `nama.csv`.")
    st.divider()

    st.subheader("📁 Sumber Data KSM")
    source_choice = st.radio(
        "Pilih Sumber Dokumen:",
        ["📂 Gunakan Dataset Lokal (dataset/)", "📤 Unggah File PDF"],
        index=0 if os.path.exists("dataset") else 1
    )

    st.divider()
    st.subheader("⏱️ Rentang Jam Kalender")
    start_hour = st.slider("Jam Mulai Kalender:", 6.0, 9.0, 7.0, 0.5)
    end_hour = st.slider("Jam Selesai Kalender:", 18.0, 22.0, 21.0, 0.5)

    st.divider()
    st.caption("🚀 OCR Cerdas: RapidOCR ONNX & PyMuPDF • Pencocokan Nama: dataset/nama.csv")


# ==========================================
# HERO BANNER
# ==========================================
st.markdown("""
<div class="main-hero">
    <h1>📚 OCR Jadwal & Assistant Schedule Manager</h1>
    <p>Ekstraksi otomatis KSM Telkom University, visualisasi Heatmap Jadwal berisikan <b>Nama Pendek Asisten</b> (sesuai nama.csv), dan Pencari Jadwal Kosong untuk koordinasi shift praktikum, piket, dan rapat.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# PROCESSING HELPER
# ==========================================
def run_processing(file_tuples):
    """file_tuples: list of (filename, file_bytes or path)"""
    engine = OCREngine()
    csv_path = "dataset/nama.csv" if os.path.exists("dataset/nama.csv") else "nama.csv"
    parser = KSMDataParser(nickname_csv_path=csv_path if os.path.exists(csv_path) else None)
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    total = len(file_tuples)
    for idx, (name, content) in enumerate(file_tuples, 1):
        status_text.text(f"Memproses ({idx}/{total}): {name}...")
        temp_path = None
        try:
            if isinstance(content, bytes):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(content)
                    temp_path = tmp.name
                target_path = temp_path
            else:
                target_path = content

            res = engine.extract_text(target_path)
            parsed = parser.parse(res)

            # Jika nama masih kosong, gunakan nama file
            if not parsed.student_info.name:
                clean_name = os.path.splitext(name)[0]
                if "-" in clean_name:
                    clean_name = clean_name.split("-")[-1].strip()
                parsed.student_info.name = clean_name

            # Pastikan nama panggilan cocok
            parser._assign_nickname(parsed.student_info)

            results.append((name, parsed))
        except Exception as e:
            logging.error(f"Error {name}: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        progress_bar.progress(idx / total)

    status_text.success(f"✅ Selesai memproses {len(results)} file KSM asisten!")
    progress_bar.empty()
    return results


# ==========================================
# DATA LOADING TRIGGER
# ==========================================
if source_choice == "📂 Gunakan Dataset Lokal (dataset/)":
    if os.path.exists("dataset"):
        dataset_files = [
            f for f in sorted(os.listdir("dataset"))
            if f.lower().endswith(".pdf")
        ]
        all_samples = []
        for f in dataset_files:
            all_samples.append((f, os.path.join("dataset", f)))

        col_src1, col_src2 = st.columns([3, 1])
        with col_src1:
            st.info(f"📂 Ditemukan **{len(all_samples)} file KSM asisten** pada direktori `dataset/` dan database nama panggilan `nama.csv`.")
        with col_src2:
            if st.button("🚀 Proses Dataset Asisten", type="primary", use_container_width=True):
                st.session_state["students_data"] = run_processing(all_samples)
    else:
        st.warning("Direktori 'dataset/' tidak ditemukan.")
else:
    uploaded = st.file_uploader(
        "Unggah file KSM Asisten (PDF):",
        type=["pdf"],
        accept_multiple_files=True
    )
    if uploaded:
        st.info(f"📎 {len(uploaded)} file siap diproses.")
        if st.button("🚀 Mulai Proses OCR KSM Asisten", type="primary"):
            file_tuples = [(f.name, f.read()) for f in uploaded]
            st.session_state["students_data"] = run_processing(file_tuples)


# ==========================================
# TABS INTERFACE
# ==========================================
tab_summary, tab_student, tab_heatmap, tab_export = st.tabs([
    "📋 Ringkasan Asisten",
    "👤 Kalender Asisten",
    "🔥 Heatmap & Jadwal Kosong Asisten",
    "💾 Export Center"
])

parsed_list = [p for _, p in st.session_state["students_data"] if p and p.courses]

# ----------------------------------------------------
# TAB 1: RINGKASAN ASISTEN
# ----------------------------------------------------
with tab_summary:
    st.subheader("📊 Hasil Pemrosesan KSM & Identitas Asisten")

    if not st.session_state["students_data"]:
        st.info("💡 Belum ada data. Silakan klik tombol 'Proses Dataset Asisten' di atas untuk memetakan KSM.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        total_files = len(st.session_state["students_data"])
        valid_students = len(parsed_list)
        digital_count = sum(1 for _, p in st.session_state["students_data"] if "words" in p.extraction_method)
        ocr_count = sum(1 for _, p in st.session_state["students_data"] if "rapidocr" in p.extraction_method)

        col1.metric("Total Dokumen", total_files)
        col2.metric("KSM Terpetakan", valid_students, f"{(valid_students/total_files)*100:.0f}%")
        col3.metric("Digital Text", digital_count)
        col4.metric("OCR (Scan/Vektor)", ocr_count)

        summary_rows = []
        for name, p in st.session_state["students_data"]:
            si = p.student_info
            has_mk = len(p.courses) > 0
            status_badge = "✅ Terpetakan" if has_mk else "⚠️ Perlu Ditinjau"
            summary_rows.append({
                "Nama Panggilan (nama.csv)": get_short_name(si),
                "Nama Lengkap Mahasiswa": si.name or "-",
                "NIM": si.nim or "-",
                "Jumlah MK": len(p.courses),
                "Total SKS": si.total_sks or p.total_calculated_sks,
                "Metode Ekstraksi": p.extraction_method,
                "Nama File": name,
                "Status": status_badge
            })

        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True)


# ----------------------------------------------------
# TAB 2: KALENDER PER ASISTEN
# ----------------------------------------------------
with tab_student:
    if not parsed_list:
        st.info("💡 Silakan proses file KSM terlebih dahulu.")
    else:
        st.subheader("👤 Detail Jadwal Perkuliahan Asisten")
        student_names = [
            f"{get_short_name(p)} — {p.student_info.name} (NIM: {p.student_info.nim or 'N/A'}) • {len(p.courses)} MK"
            for p in parsed_list
        ]
        selected_idx = st.selectbox(
            "Pilih Asisten:",
            range(len(student_names)),
            format_func=lambda i: student_names[i]
        )

        selected_student = parsed_list[selected_idx]
        si = selected_student.student_info

        # Info Box
        c_i1, c_i2, c_i3, c_i4 = st.columns(4)
        c_i1.markdown(f"**Nama Pendek:** `{get_short_name(si)}`")
        c_i2.markdown(f"**Nama Lengkap:** {si.name}")
        c_i3.markdown(f"**NIM:** {si.nim or '-'}")
        c_i4.markdown(f"**Kelas:** {si.class_id or si.cohort or '-'}")

        st.markdown(f"<span class='student-badge'>Total: {selected_student.total_calculated_sks} SKS</span> <span class='student-badge'>{selected_student.extraction_method}</span>", unsafe_allow_html=True)
        st.divider()

        # Daftar MK
        mk_rows = []
        for c in selected_student.courses:
            is_ex = is_course_excluded_from_mapping(c)
            mk_rows.append({
                "No": c.number,
                "Kode MK": c.code,
                "Nama Mata Kuliah": c.name,
                "SKS": c.sks,
                "Hari": c.schedule_day,
                "Waktu": c.schedule_time,
                "Ruangan": c.room,
                "Kelas": c.class_info,
                "Status di Heatmap": "⚪ Dikecualikan (Capstone/Magang)" if is_ex else "🟢 Aktif di Heatmap"
            })
        st.dataframe(pd.DataFrame(mk_rows), use_container_width=True)

        # Generate Visual Calendar
        mapper = CalendarMapper(start_hour=start_hour, end_hour=end_hour, output_dir="output")
        cal_img_path = mapper.create_calendar_image(
            courses=selected_student.courses,
            student_name=f"{si.name} ({get_short_name(si)})",
            nim=si.nim,
            semester=si.semester,
            filename=f"cal_{si.nim or selected_idx}.png"
        )

        st.subheader("🗓️ Visual Kalender Mingguan")
        render_full_width_image(cal_img_path)

        # Download Buttons
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            with open(cal_img_path, "rb") as f_img:
                st.download_button(
                    "📥 Unduh Kalender Visual (.png)",
                    data=f_img.read(),
                    file_name=f"Kalender_{get_short_name(si)}_{si.nim}.png",
                    mime="image/png",
                    use_container_width=True
                )
        with c_d2:
            ics_path = mapper.export_to_ics(
                courses=selected_student.courses,
                student_name=get_short_name(si),
                filename=f"cal_{si.nim or selected_idx}.ics"
            )
            with open(ics_path, "rb") as f_ics:
                st.download_button(
                    "📅 Unduh ke Google Calendar (.ics)",
                    data=f_ics.read(),
                    file_name=f"Jadwal_{get_short_name(si)}.ics",
                    mime="text/calendar",
                    use_container_width=True
                )


# ----------------------------------------------------
# TAB 3: HEATMAP & JADWAL KOSONG ASISTEN
# ----------------------------------------------------
with tab_heatmap:
    if not parsed_list:
        st.info("💡 Silakan proses file KSM terlebih dahulu.")
    else:
        st.subheader("🔥 Heatmap Jadwal & Pencari Jam Kosong Asisten")
        st.caption("Visualisasi komprehensif jadwal asisten dengan nama panggilan (berdasarkan `dataset/nama.csv`), matriks ketersediaan per jam, dan rekomendasi slot kosong untuk penugasan jaga lab/praktikum/piket.")

        # Banner & Kontrol Pengecualian MK Non-Mapping (Capstone & Magang)
        col_flt1, col_flt2 = st.columns([3, 1])
        with col_flt1:
            st.info(
                "💡 **Filter Khusus Non-Mapping Aktif:** Mata kuliah **CAPSTONE DESIGN AND PROJECT** dan "
                "**MERDEKA BELAJAR - MAGANG** secara otomatis **dikecualikan dari heatmap & pencarian jadwal kosong** "
                "karena tidak memerlukan jam kelas fisik/offline (tidak diperlukan untuk mapping penugasan asisten)."
            )
        with col_flt2:
            exclude_mapping = st.checkbox(
                "🚫 Kecualikan Capstone & Magang",
                value=True,
                help="Jika dicentang, jadwal Capstone dan MBKM Magang tidak dianggap sebagai jam sibuk kuliah di heatmap dan pencari jadwal kosong."
            )

        aggregator = ClassAggregator(start_hour=start_hour, end_hour=end_hour, exclude_non_mapping=exclude_mapping)
        aggregator.set_students(parsed_list)

        # Top summary metrics
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("👥 Total Asisten Terdata", aggregator.total_students)
        total_courses_count = len(aggregator.get_course_breakdown())
        c_m2.metric("📚 Mata Kuliah Unik", total_courses_count)
        avg_sks = sum(p.total_calculated_sks for p in parsed_list) / max(len(parsed_list), 1)
        c_m3.metric("📊 Rata-rata Beban SKS", f"{avg_sks:.1f} SKS")

        st.divider()

        # VIEW SELECTION
        view_mode = st.radio(
            "Pilih Mode Tampilan Heatmap & Penjadwalan:",
            [
                "🎯 Cari Asisten Kosong Penuh (Shift Tertentu)",
                "🟢 Heatmap Slot Mingguan (Ada Nama Asisten)",
                "👥 Heatmap Matriks per Asisten (Nama Asisten x Jam)",
                "🔍 Rekomendasi Jam Kosong Bersama (General Free Slots)"
            ],
            horizontal=True
        )

        mapper = CalendarMapper(start_hour=start_hour, end_hour=end_hour, output_dir="output")

        # =======================================================
        # SUB-MODE 0: STRICT FULL-WINDOW AVAILABILITY FINDER
        # =======================================================
        if view_mode == "🎯 Cari Asisten Kosong Penuh (Shift Tertentu)":
            st.markdown("#### 🎯 Cari Asisten yang Benar-Benar Kosong Penuh (Strict Availability)")
            st.caption(
                "Mencari asisten yang **100% bebas tanpa kuliah sedikitpun** di rentang jam yang ditentukan (misalnya **Senin–Sabtu 06:00–09:00**). "
                "Asisten yang hanya kosong di 1 jam atau 2 jam terakhir (atau memiliki kuliah di awal/tengah shift) otomatis disaring sebagai **BENTROK** "
                "disertai alasan dan rincian jam kuliahnya."
            )

            # Init session state if not set
            if "strict_start" not in st.session_state:
                st.session_state["strict_start"] = "06:00"
            if "strict_end" not in st.session_state:
                st.session_state["strict_end"] = "09:00"

            # PRESET BUTTONS
            st.markdown("##### ⚡ Preset Cepat Shift:")
            p_cols = st.columns(6)
            if p_cols[0].button("🌅 06:00 - 09:00", use_container_width=True, help="Shift Pagi Awal (Sesuai Contoh Prompt)"):
                st.session_state["strict_start"] = "06:00"
                st.session_state["strict_end"] = "09:00"
            if p_cols[1].button("☀️ 06:30 - 09:30", use_container_width=True, help="Shift Praktikum 1"):
                st.session_state["strict_start"] = "06:30"
                st.session_state["strict_end"] = "09:30"
            if p_cols[2].button("🌤️ 09:30 - 12:30", use_container_width=True, help="Shift Praktikum 2"):
                st.session_state["strict_start"] = "09:30"
                st.session_state["strict_end"] = "12:30"
            if p_cols[3].button("⛅ 12:30 - 15:30", use_container_width=True, help="Shift Praktikum 3"):
                st.session_state["strict_start"] = "12:30"
                st.session_state["strict_end"] = "15:30"
            if p_cols[4].button("🌆 15:30 - 18:30", use_container_width=True, help="Shift Praktikum 4"):
                st.session_state["strict_start"] = "15:30"
                st.session_state["strict_end"] = "18:30"
            if p_cols[5].button("🌙 18:30 - 21:00", use_container_width=True, help="Shift Malam"):
                st.session_state["strict_start"] = "18:30"
                st.session_state["strict_end"] = "21:00"

            TIME_OPTIONS_START = [f"{h:02d}:{m:02d}" for h in range(6, 21) for m in (0, 30)]
            TIME_OPTIONS_END = [f"{h:02d}:{m:02d}" for h in range(6, 22) for m in (0, 30)]

            c_f1, c_f2, c_f3 = st.columns([1, 1, 1])
            with c_f1:
                cur_start_idx = TIME_OPTIONS_START.index(st.session_state["strict_start"]) if st.session_state["strict_start"] in TIME_OPTIONS_START else 0
                chosen_start = st.selectbox("Jam Mulai Shift:", TIME_OPTIONS_START, index=cur_start_idx, key="sel_strict_start")
                st.session_state["strict_start"] = chosen_start
            with c_f2:
                cur_end_idx = TIME_OPTIONS_END.index(st.session_state["strict_end"]) if st.session_state["strict_end"] in TIME_OPTIONS_END else 6
                chosen_end = st.selectbox("Jam Selesai Shift:", TIME_OPTIONS_END, index=cur_end_idx, key="sel_strict_end")
                st.session_state["strict_end"] = chosen_end
            with c_f3:
                start_val = parse_time_float(chosen_start)
                end_val = parse_time_float(chosen_end)
                dur_calc = max(0.0, end_val - start_val)
                st.metric("Durasi Shift Penuh", f"{dur_calc:.1f} Jam")

            if start_val >= end_val:
                st.error("⚠️ Jam selesai harus lebih besar dari jam mulai.")
            else:
                days_selected = st.multiselect(
                    "Pilih Hari yang Diperiksa:",
                    ["SENIN", "SELASA", "RABU", "KAMIS", "JUMAT", "SABTU"],
                    default=["SENIN", "SELASA", "RABU", "KAMIS", "JUMAT", "SABTU"],
                    key="strict_days"
                )

                if not days_selected:
                    st.warning("Silakan pilih minimal 1 hari.")
                else:
                    report = aggregator.find_strictly_free_assistants(
                        start_time=chosen_start,
                        end_time=chosen_end,
                        days=days_selected
                    )

                    # HIGHLIGHT SUMMARY CARDS
                    st.divider()
                    h_c1, h_c2, h_c3 = st.columns(3)
                    h_c1.metric("🎯 Jam Target Shift", f"{chosen_start} - {chosen_end}", f"{dur_calc:.1f} Jam Penuh")
                    h_c2.metric("📅 Hari yang Diperiksa", f"{len(days_selected)} Hari", ", ".join(days_selected[:3]) + ("..." if len(days_selected)>3 else ""))
                    h_c3.metric("🏆 Bebas di Seluruh Hari", f"{len(report.all_days_free_assistants)} Asisten", f"{round(len(report.all_days_free_assistants)/max(aggregator.total_students,1)*100,1)}%")

                    # BANNER ASISTEN BEBAS DI SEMUA HARI
                    if report.all_days_free_assistants:
                        pills_all_free = " ".join([f"<span class='assistant-pill-free' style='font-size:0.92rem; padding:6px 14px;'>🌟 {name}</span>" for name in report.all_days_free_assistants])
                        st.markdown(f"""
                        <div style="background:#f0fdf4; border:2px solid #86efac; border-radius:12px; padding:18px 22px; margin-bottom:20px;">
                            <strong style="font-size:1.05rem; color:#166534;">
                                🏆 Asisten yang Benar-Benar Bebas Penuh di SELURUH Hari ({', '.join(days_selected)}):
                            </strong><br>
                            <span style="font-size:0.9rem; color:#14532d;">Asisten berikut 100% TIDAK MEMILIKI KULIAH di jam {chosen_start} - {chosen_end} pada SEMUA hari yang dipilih, sangat ideal untuk tugas shift harian/piket rutin:</span>
                            <div style="margin-top:10px;">{pills_all_free}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info(f"💡 Tidak ada asisten tunggal yang bebas di seluruh {len(days_selected)} hari sekaligus pada jam {chosen_start} - {chosen_end}. Silakan lihat ketersediaan per hari di bawah.")

                    # TABEL MATRIKS MINGGUAN LENGKAP
                    st.markdown("##### 📊 Matriks Status Ketersediaan Penuh Mingguan")
                    st.caption("🟢 KOSONG PENUH = Tidak ada kuliah sedikitpun • 🔴 BENTROK = Ada kuliah (lengkap dengan jam kuliahnya)")
                    st.dataframe(report.matrix_df, use_container_width=True)

                    # DETAIL PER HARI (SUB-TABS)
                    st.markdown("##### 📋 Rincian & Validasi Ketersediaan per Hari:")
                    day_tabs = st.tabs([f"📌 {d} ({len(report.per_day_results[d].completely_free_assistants)} Kosong)" for d in days_selected])

                    for idx_d, d in enumerate(days_selected):
                        with day_tabs[idx_d]:
                            d_res = report.per_day_results[d]
                            col_res1, col_res2 = st.columns([1, 1])

                            with col_res1:
                                st.markdown(f"###### 🟢 Asisten Benar-benar Kosong ({len(d_res.completely_free_assistants)} Orang / {d_res.free_percentage}%)")
                                if d_res.completely_free_assistants:
                                    free_html = " ".join([f"<span class='assistant-pill-free'>✓ {name}</span>" for name in d_res.completely_free_assistants])
                                    st.markdown(free_html, unsafe_allow_html=True)
                                    st.caption(f"Semua {len(d_res.completely_free_assistants)} asisten di atas 100% bebas dari jam {chosen_start} hingga {chosen_end}.")
                                else:
                                    st.warning(f"Tidak ada asisten yang kosong penuh pada hari {d} di jam {chosen_start} - {chosen_end}.")

                            with col_res2:
                                st.markdown(f"###### 🔴 Asisten Tidak Kosong Penuh / Bentrok ({len(d_res.busy_assistants)} Orang)")
                                if d_res.busy_assistants:
                                    b_rows = []
                                    for b in d_res.busy_assistants:
                                        for c in b.conflicts:
                                            b_rows.append({
                                                "Nama": b.name,
                                                "Mata Kuliah": c.course_name,
                                                "Jadwal Kuliah": c.schedule_time,
                                                "Alasan / Mengapa Tidak Lolos": c.explanation
                                            })
                                    st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)
                                else:
                                    st.success(f"🎉 Tidak ada yang bentrok pada hari {d}! Seluruh asisten bebas penuh.")

                    # EXPORT EXCEL BUTTON
                    st.markdown("---")
                    col_dl1, col_dl2 = st.columns([2, 1])
                    with col_dl1:
                        st.markdown("###### 💾 Unduh Rekap Hasil Pencarian Shift")
                        st.caption("Simpan matriks mingguan dan rincian asisten kosong penuh ke dalam format Excel (.xlsx).")
                    with col_dl2:
                        excel_path = f"output/Asisten_Kosong_Penuh_{chosen_start.replace(':','')}_{chosen_end.replace(':','')}.xlsx"
                        aggregator.export_strict_report_to_excel(report, excel_path)
                        with open(excel_path, "rb") as f_ex:
                            st.download_button(
                                "📥 Unduh Laporan Excel (.xlsx)",
                                data=f_ex.read(),
                                file_name=f"Asisten_Kosong_Penuh_{chosen_start.replace(':','')}_{chosen_end.replace(':','')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )

        # =======================================================
        # SUB-MODE 1: HEATMAP SLOT MINGGUAN (ADA NAMA ASISTEN)
        # =======================================================
        elif view_mode == "🟢 Heatmap Slot Mingguan (Ada Nama Asisten)":
            st.markdown("#### 🗓️ Heatmap Kepadatan & Ketersediaan Mingguan")
            
            c_h1, c_h2 = st.columns([2, 1])
            with c_h1:
                heat_target = st.radio(
                    "Fokus Tampilan Heatmap:",
                    ["🟢 Jadwal Kosong / Ketersediaan Asisten (Free Slots)", "🔴 Jadwal Kuliah / Kesibukan Asisten (Busy Slots)"],
                    horizontal=True
                )
            with c_h2:
                st.caption("Setiap sel menampilkan jumlah asisten beserta **daftar nama pendek asisten** yang sesuai.")

            is_free_mode = "Kosong" in heat_target

            if is_free_mode:
                matrix_df = aggregator.build_free_matrix()
                details_dict = aggregator._free_details
                heatmap_img = mapper.create_class_heatmap_image(
                    matrix_df=matrix_df,
                    details_dict=details_dict,
                    mode="free",
                    class_name="Asisten Laboratorium",
                    total_students=aggregator.total_students,
                    filename="heatmap_asisten_kosong.png"
                )
            else:
                matrix_df = aggregator.build_occupancy_matrix()
                details_dict = aggregator._matrix_details
                heatmap_img = mapper.create_class_heatmap_image(
                    matrix_df=matrix_df,
                    details_dict=details_dict,
                    mode="busy",
                    class_name="Asisten Laboratorium",
                    total_students=aggregator.total_students,
                    filename="heatmap_asisten_kuliah.png"
                )

            render_full_width_image(heatmap_img)

            with open(heatmap_img, "rb") as f_hm:
                st.download_button(
                    f"📥 Unduh Heatmap PNG ({'Jadwal Kosong' if is_free_mode else 'Jadwal Kuliah'})",
                    data=f_hm.read(),
                    file_name=f"Heatmap_Asisten_{'Kosong' if is_free_mode else 'Kuliah'}.png",
                    mime="image/png"
                )

            # INTERACTIVE SLOT INSPECTOR
            st.markdown("---")
            st.markdown("#### 🔎 Inspeksi Detail Slot Waktu & Nama Asisten")
            st.caption("Pilih hari dan slot waktu untuk melihat secara lengkap siapa saja asisten yang **BEBAS (KOSONG)** dan siapa yang **KULIAH**.")

            insp_c1, insp_c2 = st.columns(2)
            with insp_c1:
                insp_day = st.selectbox("Pilih Hari:", ["SENIN", "SELASA", "RABU", "KAMIS", "JUMAT", "SABTU"], key="insp_day")
            with insp_c2:
                time_slot_labels = list(matrix_df.index)
                insp_slot = st.selectbox("Pilih Slot Waktu:", range(len(time_slot_labels)), format_func=lambda i: time_slot_labels[i], key="insp_slot")

            free_assts, busy_assts, busy_courses = aggregator.get_slot_assistant_details(insp_day, insp_slot)

            detail_c1, detail_c2 = st.columns(2)
            with detail_c1:
                st.markdown(f"##### 🟢 Asisten Bebas / Kosong ({len(free_assts)} Orang)")
                if free_assts:
                    pills_html = " ".join([f"<span class='assistant-pill-free'>✓ {name}</span>" for name in free_assts])
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.warning("Tidak ada asisten yang kosong di jam ini (semua ada kelas).")

            with detail_c2:
                st.markdown(f"##### 🔴 Asisten Sedang Kuliah ({len(busy_assts)} Orang)")
                if busy_assts:
                    busy_rows = []
                    for name in busy_assts:
                        c_info = busy_courses.get(name, "Ada Kuliah")
                        busy_rows.append({"Nama Panggilan": name, "Mata Kuliah / Ruangan": c_info})
                    st.dataframe(pd.DataFrame(busy_rows), use_container_width=True, hide_index=True)
                else:
                    st.success("🎉 Seluruh asisten bebas! Tidak ada yang sedang kuliah di jam ini.")

        # =======================================================
        # SUB-MODE 2: HEATMAP MATRIKS PER ASISTEN (NAMA X JAM)
        # =======================================================
        elif view_mode == "👥 Heatmap Matriks per Asisten (Nama Asisten x Jam)":
            st.markdown("#### 👥 Matriks Ketersediaan Seluruh Asisten per Jam")
            st.caption("Sumbu Y memuat **seluruh nama pendek asisten** dari `nama.csv`. Setiap kotak menunjukkan secara langsung apakah asisten tersebut **KOSONG (Hijau)** atau **KULIAH (Merah)**.")

            day_choice = st.selectbox(
                "Pilih Hari untuk Matriks Asisten:",
                ["SENIN", "SELASA", "RABU", "KAMIS", "JUMAT", "SABTU"],
                key="mat_day"
            )

            text_df, num_df = aggregator.build_assistant_schedule_matrix(day=day_choice)

            matrix_img = mapper.create_assistant_matrix_heatmap_image(
                num_df=num_df,
                text_df=text_df,
                day_name=day_choice,
                filename=f"assistant_matrix_{day_choice.lower()}.png"
            )

            render_full_width_image(matrix_img)

            with open(matrix_img, "rb") as f_mat:
                st.download_button(
                    f"📥 Unduh Matriks Asisten Hari {day_choice} (.png)",
                    data=f_mat.read(),
                    file_name=f"Matriks_Asisten_{day_choice}.png",
                    mime="image/png"
                )

            with st.expander(f"📋 Lihat Tabel Data Mentah Ketersediaan ({day_choice})"):
                st.dataframe(text_df, use_container_width=True)

        # =======================================================
        # SUB-MODE 3: SMART FREE SCHEDULE FINDER
        # =======================================================
        elif view_mode == "🔍 Cari Jadwal Kosong Asisten (Free Slot Finder)":
            st.markdown("#### 🔍 Rekomendasi Jam Kosong Bersama (Free Slot Finder)")
            st.caption("Cari jadwal kosong terbaik untuk penugasan jaga lab, asistensi praktikum, piket, atau rapat asisten.")

            f_c1, f_c2, f_c3 = st.columns(3)
            with f_c1:
                filter_day = st.selectbox(
                    "Filter Hari:",
                    ["SEMUA HARI", "SENIN", "SELASA", "RABU", "KAMIS", "JUMAT", "SABTU"]
                )
            with f_c2:
                tolerance = st.slider("Toleransi Asisten Berhalangan (Ada Kelas):", 0, 8, 2)
            with f_c3:
                min_dur = st.slider("Minimal Durasi Jam Kosong:", 1.0, 4.0, 1.5, 0.5)

            selected_day_filter = None if filter_day == "SEMUA HARI" else filter_day
            free_slots = aggregator.find_free_slots(
                min_duration_hours=min_dur,
                max_busy_students=tolerance,
                day_filter=selected_day_filter
            )

            if not free_slots:
                st.warning("⚠️ Tidak ditemukan slot kosong dengan kriteria tersebut. Coba naikkan toleransi asisten berhalangan atau kurangi durasi minimal.")
            else:
                st.success(f"✨ Ditemukan **{len(free_slots)} slot jadwal kosong** yang memenuhi kriteria!")

                for idx, fs in enumerate(free_slots[:10], 1):
                    free_names_str = ", ".join(fs.free_student_names)
                    busy_names_str = ", ".join(fs.busy_student_names) if fs.busy_student_names else "Semua Bebas"

                    with st.container():
                        st.markdown(f"""
                        <div class="free-slot-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <strong style="font-size:1.1rem; color:#0f172a;">
                                    📌 Slot #{idx}: {fs.day}, {fs.start_time} - {fs.end_time} ({fs.duration_hours} Jam)
                                </strong>
                                <span class="nickname-badge">{fs.free_percentage}% Asisten Bebas ({fs.free_student_count}/{fs.total_students})</span>
                            </div>
                            <div style="margin-top:6px;">
                                <span style="font-size:0.88rem; font-weight:700; color:#15803d;">👥 Asisten Kosong ({fs.free_student_count} orang):</span><br>
                                <span style="font-size:0.85rem; color:#1e293b;">{free_names_str}</span>
                            </div>
                            <div style="margin-top:8px;">
                                <span style="font-size:0.88rem; font-weight:700; color:#b91c1c;">⚠️ Berhalangan Kuliah ({len(fs.busy_student_names)} orang):</span>
                                <span style="font-size:0.85rem; color:#475569;"> {busy_names_str}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Rekapitulasi Tabel Slot Kosong
                with st.expander("📊 Rekapitulasi Seluruh Slot Kosong dalam Bentuk Tabel"):
                    table_rows = []
                    for fs in free_slots:
                        table_rows.append({
                            "Hari": fs.day,
                            "Jam Mulai": fs.start_time,
                            "Jam Selesai": fs.end_time,
                            "Durasi (Jam)": fs.duration_hours,
                            "Jumlah Kosong": f"{fs.free_student_count}/{fs.total_students}",
                            "Persentase": f"{fs.free_percentage}%",
                            "Daftar Asisten Kosong": ", ".join(fs.free_student_names),
                            "Asisten Berhalangan": ", ".join(fs.busy_student_names)
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # Sebaran Mata Kuliah yang Diambil Kelas
        st.divider()
        st.markdown("#### 📚 Sebaran Mata Kuliah yang Diambil Asisten")
        st.caption("Daftar mata kuliah yang diambil asisten beserta nama pendek peserta untuk mengantisipasi jadwal bentrok.")
        df_cb = aggregator.get_course_breakdown()
        st.dataframe(df_cb, use_container_width=True)


# ----------------------------------------------------
# TAB 4: EXPORT CENTER
# ----------------------------------------------------
with tab_export:
    st.subheader("💾 Pusat Unduhan & Ekspor Data")
    if not parsed_list:
        st.info("💡 Belum ada data untuk diekspor.")
    else:
        st.markdown("Unduh rekapitulasi lengkap dalam berbagai format standar dengan **nama pendek asisten**:")

        aggregator = ClassAggregator(start_hour=start_hour, end_hour=end_hour)
        aggregator.set_students(parsed_list)

        col_ex1, col_ex2 = st.columns(2)

        # 1. Excel Export
        with col_ex1:
            st.markdown("##### 📊 Rekap Asisten & Jadwal Kosong (.xlsx)")
            st.caption("Memuat 5 sheet terstruktur: Daftar Asisten (Nama Pendek & Lengkap), Matriks Asisten Kosong, Matriks Asisten Kuliah, Rekomendasi Jam Kosong, dan Rekap Mata Kuliah.")
            excel_path = "output/Rekap_Jadwal_Asisten.xlsx"
            aggregator.export_to_excel(excel_path)
            with open(excel_path, "rb") as f_ex:
                st.download_button(
                    "📥 Unduh Rekap Excel (.xlsx)",
                    data=f_ex.read(),
                    file_name=f"Rekap_Jadwal_Asisten_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        # 2. Bulk ICS Zip
        with col_ex2:
            st.markdown("##### 📅 Paket Kalender Google (.ics ZIP)")
            st.caption("Kumpulan file .ics seluruh asisten dalam satu file arsip ZIP untuk langsung diimpor ke Google Calendar.")
            zip_buffer = io.BytesIO()
            mapper = CalendarMapper(output_dir="output")
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for p in parsed_list:
                    si = p.student_info
                    ics_content_path = mapper.export_to_ics(p.courses, student_name=get_short_name(si), filename=f"{si.nim or 'asisten'}.ics")
                    zf.write(ics_content_path, arcname=f"Jadwal_{get_short_name(si)}_{si.nim}.ics")

            st.download_button(
                "📥 Unduh Seluruh Kalender (.zip)",
                data=zip_buffer.getvalue(),
                file_name=f"Kalender_Asisten_{datetime.now().strftime('%Y%m%d')}.zip",
                mime="application/zip",
                use_container_width=True
            )
