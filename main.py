#!/usr/bin/env python3
"""
Aplikasi OCR Process Mapping KSM - Batch & Per-Student Calendar Generator
Mendukung pemrosesan perorangan dan batch 22-34 KSM (satu kelas).
"""

import argparse
import sys
import os
import logging
import json
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ocr_engine import OCREngine, OCRResult
from data_parser import KSMDataParser, KSMParsedData, Course, StudentInfo
from calendar_mapper import CalendarMapper
from class_aggregator import ClassAggregator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("Main")


def print_banner():
    print("=" * 65)
    print("  📚 KSM OCR Process Mapping & Class Calendar Generator")
    print("  Pemetaan Jadwal KSM Telkom University (Perorangan & Batch Kelas)")
    print("=" * 65)
    print()


def print_student_summary(parsed_data: KSMParsedData, output_files: list):
    """Cetak ringkasan hasil parsing mahasiswa."""
    print("\n" + "=" * 65)
    print("  RINGKASAN HASIL PER MAHASISWA")
    print("=" * 65)

    si = parsed_data.student_info
    print(f"\n  📋 Mahasiswa    : {si.name}")
    print(f"  🆔 NIM         : {si.nim}")
    print(f"  🎓 Program Studi: {si.program_study}")
    print(f"  📅 Semester     : {si.semester}")
    print(f"  🏫 Angkatan/Kelas: {si.cohort} / {si.class_id}")
    print(f"  👤 Dosen Wali  : {si.advisor}")
    print(f"  📊 Total SKS    : {si.total_sks or parsed_data.total_calculated_sks}")
    print(f"  ⚙️  Metode OCR  : {parsed_data.extraction_method}")

    print(f"\n  📚 Daftar Mata Kuliah ({len(parsed_data.courses)} MK):")
    print("  " + "-" * 75)
    print(f"  {'No.':<4} {'Kode':<10} {'Nama MK':<32} {'SKS':<4} {'Jadwal':<24}")
    print("  " + "-" * 75)

    for c in parsed_data.courses:
        print(
            f"  {c.number:<4} {c.code:<10} {c.name[:30]:<32} "
            f"{c.sks:<4} {c.schedule_day} {c.schedule_time}"
        )

    print("  " + "-" * 75)
    print(f"\n  📁 File yang dihasilkan:")
    for f in output_files:
        print(f"     → {f}")
    print()


def process_single_ksm(
    pdf_path: str,
    output_dir: str = "output",
    output_name: str = None,
    output_format: str = "both"
) -> Tuple[KSMParsedData, list]:
    if not output_name:
        output_name = os.path.splitext(os.path.basename(pdf_path))[0]

    os.makedirs(output_dir, exist_ok=True)
    engine = OCREngine()
    parser = KSMDataParser()
    mapper = CalendarMapper(output_dir=output_dir)

    result = engine.extract_text(pdf_path=pdf_path)
    parsed_data = parser.parse(result)

    if not parsed_data.courses:
        logger.warning(f"Tidak ada mata kuliah yang terdeteksi dari: {pdf_path}")
        return parsed_data, []

    output_files = []
    si = parsed_data.student_info

    # 1. Gambar Kalender PNG
    if output_format in ("png", "both"):
        png_path = mapper.create_calendar_image(
            courses=parsed_data.courses,
            student_name=si.name,
            nim=si.nim,
            semester=si.semester,
            filename=f"{output_name}_calendar.png"
        )
        output_files.append(png_path)

    # 2. Kalender iCalendar (.ics)
    if output_format in ("ics", "both"):
        ics_path = mapper.export_to_ics(
            courses=parsed_data.courses,
            student_name=si.name,
            filename=f"{output_name}.ics"
        )
        output_files.append(ics_path)

    # 3. Metadata JSON
    json_path = os.path.join(output_dir, f"{output_name}_metadata.json")
    metadata = {
        "student_info": {
            "name": si.name, "nim": si.nim,
            "program_study": si.program_study,
            "semester": si.semester,
            "cohort": si.cohort, "class_id": si.class_id,
            "advisor": si.advisor, "total_sks": si.total_sks,
        },
        "courses": [
            {
                "number": c.number, "code": c.code, "name": c.name,
                "sks": c.sks, "class_info": c.class_info,
                "schedule_day": c.schedule_day,
                "schedule_time": c.schedule_time,
                "room": c.room,
            }
            for c in parsed_data.courses
        ],
        "source_file": pdf_path,
        "extraction_method": parsed_data.extraction_method,
        "document_date": parsed_data.document_date,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    output_files.append(json_path)

    return parsed_data, output_files


def process_batch_folder(pdf_dir: str, output_dir: str = "output", output_format: str = "both"):
    """Proses batch 22-34 file KSM dalam satu direktori dan agregasi kelas."""
    pdf_files = [
        f for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print(f"❌ Tidak ada file PDF ditemukan di {pdf_dir}")
        return

    print(f"🔍 Menemukan {len(pdf_files)} file PDF di {pdf_dir}")
    aggregator = ClassAggregator()
    mapper = CalendarMapper(output_dir=output_dir)

    all_parsed_students = []
    generated_files = []

    print("\n[1/3] ⚙️  Mengekstraksi teks & jadwal setiap KSM...")
    for idx, pdf_file in enumerate(sorted(pdf_files), 1):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        try:
            parsed_data, out_files = process_single_ksm(
                pdf_path=pdf_path,
                output_dir=output_dir,
                output_format=output_format
            )
            if parsed_data.courses:
                all_parsed_students.append(parsed_data)
                aggregator.add_student(parsed_data)
                generated_files.extend(out_files)
                print(f"  [{idx:2d}/{len(pdf_files):2d}] ✅ {pdf_file[:35]:<35} ({len(parsed_data.courses)} MK, {parsed_data.extraction_method})")
            else:
                print(f"  [{idx:2d}/{len(pdf_files):2d}] ⚠️  {pdf_file[:35]:<35} (Lewati: Tidak ada tabel MK/Link)")
        except Exception as e:
            logger.error(f"Gagal memproses {pdf_file}: {e}")

    total_valid = len(all_parsed_students)
    print(f"\n[2/3] 📊 Menganalisis agregasi kelas ({total_valid} mahasiswa berhasil dimuat)...")

    # Matriks Occupancy
    df_matrix = aggregator.build_occupancy_matrix()

    # Rekomendasi Jam Kosong
    free_slots = aggregator.find_free_slots(min_duration_hours=1.0, max_busy_students=2)

    print("\n" + "=" * 65)
    print("  🗓️  REKOMENDASI JAM KOSONG BERSAMA (FREE SLOT FINDER)")
    print("=" * 65)
    for fs in free_slots[:6]:
        print(f"  ✨ {fs.day:<7} {fs.start_time}-{fs.end_time} ({fs.duration_hours} Jam) | Kosong: {fs.free_percentage}% ({fs.free_student_count}/{fs.total_students} Mahasiswa)")
    print("=" * 65)

    print("\n[3/3] 💾 Menghasilkan file rekap kelas...")
    # 1. Excel Rekap Kelas
    excel_path = os.path.join(output_dir, "Rekap_Jadwal_Kelas.xlsx")
    aggregator.export_to_excel(excel_path)
    generated_files.append(excel_path)

    # 2. Visual Heatmap Kelas
    heatmap_path = mapper.create_class_heatmap_image(
        matrix_df=df_matrix,
        class_name="Kelas Mahasiswa",
        total_students=total_valid,
        filename="Class_Schedule_Heatmap.png"
    )
    generated_files.append(heatmap_path)

    print(f"\n🎉 SELESAI! {len(generated_files)} file telah dibuat di direktori '{output_dir}':")
    print(f"   📊 Rekap Excel : {excel_path}")
    print(f"   🔥 Heatmap PNG : {heatmap_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="OCR Process Mapping KSM - Batch & Per-Student Calendar Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Contoh Penggunaan:\n"
               "  1. Proses file tunggal:\n"
               "     python main.py \"KSM_Andikanajmi Levi Maheswara.pdf\"\n"
               "  2. Proses batch 22-34 file KSM:\n"
               "     python main.py dataset/ --batch\n"
    )
    parser.add_argument("input", help="Path file PDF KSM atau folder dataset (jika --batch)")
    parser.add_argument("--format", "-f", choices=["png", "ics", "both"], default="both")
    parser.add_argument("--output-dir", "-o", default="output", help="Direktori output (default: output/)")
    parser.add_argument("--batch", "-b", action="store_true", help="Mode batch pemrosesan satu folder KSM")

    args = parser.parse_args()
    print_banner()

    if args.batch or os.path.isdir(args.input):
        process_batch_folder(args.input, output_dir=args.output_dir, output_format=args.format)
    else:
        if not os.path.exists(args.input):
            print(f"❌ File tidak ditemukan: {args.input}")
            sys.exit(1)

        parsed_data, output_files = process_single_ksm(
            pdf_path=args.input,
            output_dir=args.output_dir,
            output_format=args.format
        )
        if parsed_data.courses:
            print_student_summary(parsed_data, output_files)


if __name__ == "__main__":
    main()