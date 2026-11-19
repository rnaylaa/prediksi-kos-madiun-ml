import os
import pandas as pd
import plotly.express as px
import json
from django.shortcuts import render
from . import ml

# Ambil API key dari file .env / environment variable (Aman untuk GitHub)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def rupiah(ribu):
    return "Rp " + f"{ribu * 1000:,.0f}".replace(",", ".")


def _dapatkan_ai_insights(df_top5, input_user=None):
    """Menghasilkan analisis AI yang dinamis dan spesifik membaca data/grafik yang sedang tampil."""
    if df_top5.empty:
        return {
            "hist": "Tidak ada data kos yang sesuai dengan filter.",
            "bar": "Tidak ada data untuk dibandingkan.",
            "scatter_jarak": "Data sebaran kosong.",
            "scatter_rating": "Data rating kosong.",
        }

    # Hitung poin-poin statistik utama dari grafik yang sedang tampil
    kos_min = df_top5.loc[df_top5["harga"].idxmin()]
    kos_max = df_top5.loc[df_top5["harga"].idxmax()]
    kos_dekat = df_top5.loc[df_top5["jarak_km"].idxmin()]
    kos_best = df_top5.loc[df_top5["rating"].idxmax()]

    # Fallback dinamis (otomatis membaca data asli jika API offline)
    default_insights = {
        "hist": f"Rentang harga kos hasil filter berada di antara {rupiah(kos_min['harga'])} ({kos_min['nama']}) hingga {rupiah(kos_max['harga'])} ({kos_max['nama']}).",
        "bar": f"Pada grafik batang, {kos_max['nama']} memiliki tarif tertinggi ({rupiah(kos_max['harga'])}), sedangkan {kos_min['nama']} paling ekonomis ({rupiah(kos_min['harga'])}).",
        "scatter_jarak": f"Pilihan terdekat ke kampus adalah {kos_dekat['nama']} ({kos_dekat['jarak_km']} km) dengan sewa {rupiah(kos_dekat['harga'])}/bulan.",
        "scatter_rating": f"Kepuasan tertinggi diraih oleh {kos_best['nama']} dengan rating ⭐ {kos_best['rating']} pada harga {rupiah(kos_best['harga'])}.",
    }

    if (
        not GROQ_API_KEY
        or GROQ_API_KEY == ""
    ):
        return default_insights

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)

        # Rangkum data 5 kos yang sedang divisualisasikan
        ringkasan = []
        for _, row in df_top5.iterrows():
            ringkasan.append(
                f"- {row['nama']}: Jarak {row['jarak_km']} km, AC {row['ac'].capitalize()}, Rating {row['rating']}, Harga Rp {int(row['harga'] * 1000)}"
            )
        data_text = "\n".join(ringkasan)

        prompt = f"""
        Anda adalah data analyst properti kos mahasiswa di Madiun.
        Berikut adalah data 5 kos yang SEDANG TAMPIL di grafik saat ini:
        {data_text}

        Filter yang dipilih user: Jarak maks {input_user['jarak']} km, AC: {input_user['ac']}, Budget maks: Rp {int(input_user['budget'] * 1000)}.

        Tugas Anda: Buat interpretasi/pembacaan grafik yang SPESIFIK dan BERBEDA berdasarkan data di atas (1 kalimat jelas per grafik, sebutkan nama kos dan nominal/angkanya):
        1. hist: Jelaskan sebaran dan rentang harga dari kos termurah sampai termahal yang tampak di histogram.
        2. bar: Baca perbandingan tinggi batang harga kos (kos mana yang paling mahal vs paling hemat).
        3. scatter_jarak: Baca titik sebaran jarak vs harga (apakah kos terdekat harganya terjangkau).
        4. scatter_rating: Baca titik rating vs harga (apakah kos dengan bintang tertinggi sebanding harganya).

        Kembalikan HANYA JSON format berikut:
        {{
            "hist": "penjelasan grafik histogram...",
            "bar": "penjelasan grafik batang...",
            "scatter_jarak": "penjelasan grafik scatter jarak vs harga...",
            "scatter_rating": "penjelasan grafik scatter rating vs harga..."
        }}
        """

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            temperature=0.6,  # Membuat narasinya luwes dan bervariasi setiap input berubah
        )

        return json.loads(chat_completion.choices[0].message.content)

    except Exception as e:
        print(f"Error memanggil Groq API: {e}")
        return default_insights

    # Jika API key belum diganti, gunakan insight default agar web tidak error
    if (
        not GROQ_API_KEY
        or GROQ_API_KEY == ""
    ):
        return default_insights

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)

        # Buat ringkasan data teks untuk dikirim ke Groq
        ringkasan_data = []
        for _, row in df_top5.iterrows():
            ringkasan_data.append(
                f"- {row['nama']}: Jarak {row['jarak_km']} km, Rating {row['rating']}, Harga Rp {int(row['harga'] * 1000)}"
            )
        data_text = "\n".join(ringkasan_data)

        prompt = f"""
        Anda adalah asisten data analyst properti kos mahasiswa di Madiun.
        Berikut adalah data 5 kos pilihan teratas:
        {data_text}

        Berikan analisis cerdas dan ringkas (1 kalimat padat dan informatif per grafik) dalam bahasa Indonesia untuk 4 grafik berikut:
        1. hist: Insight tentang distribusi/rentang harga kos tersebut.
        2. bar: Perbandingan harga dan jarak kos mana yang paling mahal/murah.
        3. scatter_jarak: Pola hubungan jarak vs harga (apakah lebih dekat lebih mahal).
        4. scatter_rating: Analisis nilai value-for-money berdasarkan rating vs harga.

        KEMBALIKAN HANYA FORMAT JSON MURNI seperti ini:
        {{
            "hist": "1 kalimat penjelasan grafik distribusi harga...",
            "bar": "1 kalimat penjelasan perbandingan harga vs jarak...",
            "scatter_jarak": "1 kalimat penjelasan hubungan jarak dan harga...",
            "scatter_rating": "1 kalimat penjelasan rating vs harga..."
        }}
        """

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        response_content = chat_completion.choices[0].message.content
        return json.loads(response_content)

    except Exception as e:
        print(f"Error memanggil Groq API: {e}")
        return default_insights


def _buat_charts(df_top5):
    """Membuat 4 grafik khusus 5 kos teratas untuk layout Grid 2x2."""
    df_plot = df_top5.copy()
    df_plot["harga_rp"] = df_plot["harga"].apply(rupiah)

    common_layout = dict(
        height=320,  # Disesuaikan agar muat dengan kotak AI Insight di bawahnya
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            color="#141A15",
        ),
        xaxis=dict(
            gridcolor="#E5E7EB",
            linecolor="#E5E7EB",
            tickfont=dict(color="#6E746F", size=10),
            title=dict(font=dict(color="#141A15", size=11)),
        ),
        yaxis=dict(
            gridcolor="#E5E7EB",
            linecolor="#E5E7EB",
            tickfont=dict(color="#6E746F", size=10),
            title=dict(font=dict(color="#141A15", size=11)),
        ),
        hoverlabel=dict(
            bgcolor="#141A15",
            font_color="#FFFFFF",
            font_family="Inter, sans-serif",
        ),
        margin=dict(l=35, r=15, t=45, b=35),
    )

    # 1. Histogram: Distribusi Harga
    fig_hist = px.histogram(
        df_plot,
        x="harga",
        nbins=6,
        color="nama",
        color_discrete_sequence=[
            "#141A15",
            "#4D7C0F",
            "#84CC16",
            "#9EF01A",
            "#A3E635",
        ],
        title="Distribusi Harga (5 Kos Teratas)",
        labels={"harga": "Harga (ribu)", "nama": "Kos"},
    )
    fig_hist.update_layout(**common_layout, bargap=0.15, showlegend=False)

    # 2. Batang: Harga vs Jarak
    fig_bar = px.bar(
        df_plot.sort_values(by="jarak_km"),
        x="nama",
        y="harga",
        color="harga",
        color_continuous_scale=[[0, "#E2E8F0"], [1, "#9EF01A"]],
        text="harga",
        hover_data=["jarak_km", "rating"],
        title="Harga Menurut Jarak (km)",
        labels={
            "nama": "Nama Kos",
            "harga": "Harga (ribu)",
            "jarak_km": "Jarak (km)",
        },
    )
    fig_bar.update_traces(
        textposition="outside",
        marker_line_color="#141A15",
        marker_line_width=1,
        textfont=dict(color="#141A15", size=11),
    )
    fig_bar.update_layout(**common_layout, coloraxis_showscale=False)

    # 3. Scatter 1: Jarak vs Harga
    fig_scatter_jarak = px.scatter(
        df_plot,
        x="jarak_km",
        y="harga",
        hover_name="nama",
        hover_data=["rating"],
        text="nama",
        title="Scatter: Jarak (km) vs Harga",
        labels={"jarak_km": "Jarak (km)", "harga": "Harga (ribu)"},
    )
    fig_scatter_jarak.update_traces(
        marker=dict(color="#9EF01A", size=20, line=dict(width=1.5, color="#141A15")),
        textposition="top center",
        textfont=dict(color="#141A15", size=10),
    )
    fig_scatter_jarak.update_layout(**common_layout, showlegend=False)

    # 4. Scatter 2: Rating vs Harga
    fig_scatter_rating = px.scatter(
        df_plot,
        x="rating",
        y="harga",
        hover_name="nama",
        hover_data=["jarak_km"],
        text="nama",
        title="Scatter: Rating vs Harga",
        labels={"rating": "Rating", "harga": "Harga (ribu)"},
    )
    fig_scatter_rating.update_traces(
        marker=dict(color="#9EF01A", size=20, line=dict(width=1.5, color="#141A15")),
        textposition="top center",
        textfont=dict(color="#141A15", size=10),
    )
    fig_scatter_rating.update_layout(**common_layout, showlegend=False)

    figs = [fig_hist, fig_bar, fig_scatter_jarak, fig_scatter_rating]

    return [
        f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        for i, f in enumerate(figs)
    ]


def _dapatkan_rekomendasi(df, input_user, limit=5):
    """Filter kos dari dataset berdasarkan fasilitas AC, budget, dan jarak terdekat."""
    kondisi_ac = df["ac"] == input_user["ac"]
    kondisi_budget = df["harga"] <= input_user["budget"]
    target_jarak = input_user["jarak"]
    kondisi_jarak = df["jarak_km"] <= (target_jarak + 0.5)

    hasil_filter = df[kondisi_ac & kondisi_budget & kondisi_jarak].copy()

    if len(hasil_filter) < 3:
        hasil_filter = df[kondisi_ac & kondisi_budget].copy()
        if not hasil_filter.empty:
            hasil_filter["selisih"] = (
                hasil_filter["jarak_km"] - target_jarak
            ).abs()
            hasil_filter = hasil_filter.sort_values(
                by=["selisih", "rating"], ascending=[True, False]
            )

    if hasil_filter.empty:
        hasil_filter = (
            df[kondisi_ac].sort_values(by=["harga", "rating"]).head(limit)
        )
    else:
        hasil_filter = hasil_filter.sort_values(
            by=["rating", "jarak_km"], ascending=[False, True]
        ).head(limit)

    df_top5 = hasil_filter.copy()

    rekomendasi_rows = [
        [
            row["nama"],
            f"{row['jarak_km']} km",
            row["ac"].capitalize(),
            f"⭐ {row['rating']}",
            rupiah(row["harga"]),
        ]
        for _, row in df_top5.iterrows()
    ]

    return df_top5, rekomendasi_rows


def index(request):
    hasil = None
    input_user = {"jarak": 2.0, "ac": "ya", "rating": 4.0, "budget": 1200}

    if request.method == "POST":
        input_user = {
            "jarak": float(request.POST["jarak"]),
            "ac": request.POST.get("ac", "ya"),
            "rating": float(request.POST["rating"]),
            "budget": float(request.POST.get("budget", 1200)),
        }
        prediksi = max(
            ml.prediksi(
                ml.MODEL,
                jarak=input_user["jarak"],
                ac=input_user["ac"],
                rating=input_user["rating"],
            ),
            0,
        )
        hasil = rupiah(prediksi)

    df_top5, rekomendasi_rows = _dapatkan_rekomendasi(ml.DF, input_user)

    # 1. Buat 4 grafik plotly
    charts = _buat_charts(df_top5)

    # 2. Dapatkan AI Insight untuk masing-masing grafik dari Groq
    ai_insights = _dapatkan_ai_insights(df_top5, input_user)

    # Pasangkan setiap chart dengan insight AI-nya
    charts_with_ai = [
        {"chart": charts[0], "insight": ai_insights.get("hist", "")},
        {"chart": charts[1], "insight": ai_insights.get("bar", "")},
        {"chart": charts[2], "insight": ai_insights.get("scatter_jarak", "")},
        {"chart": charts[3], "insight": ai_insights.get("scatter_rating", "")},
    ]

    df_all = ml.DF.copy()
    df_all["harga_rp"] = df_all["harga"].apply(rupiah)
    tabel_headers = ["Nama Kos", "Jarak (km)", "AC", "Rating", "Harga/Bulan"]
    tabel_rows = [
        [
            row["nama"],
            row["jarak_km"],
            row["ac"].capitalize(),
            row["rating"],
            row["harga_rp"],
        ]
        for _, row in df_all.iterrows()
    ]

    context = {
        "hasil": hasil,
        "input": input_user,
        "budget_rp": rupiah(input_user["budget"]),
        "mae": rupiah(ml.MAE),
        "r2": round(ml.R2, 2),
        "peringatan": ml.R2 < 0,
        "charts_with_ai": charts_with_ai,
        "tabel_headers": tabel_headers,
        "tabel_rows": tabel_rows,
        "rekomendasi_rows": rekomendasi_rows,
    }
    return render(request, "prediksi/index.html", context)