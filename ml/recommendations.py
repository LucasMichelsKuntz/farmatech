from dataclasses import dataclass

from ml.enums import Priority

OPTIMAL_RANGES = {
    "nitrogenio":  (20, 100),
    "fosforo":     (20, 100),
    "potassio":    (20, 100),
    "ph":          (5.5, 7.5),
    "umidade":     (40, 85),
    "temperatura": (15, 35),
}

_SORT_ORDER = [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.OK]


@dataclass
class Recommendation:
    priority:      Priority
    parameter:     str
    current_value: float
    optimal_range: str
    action:        str
    impact_pct:    float = 0.0


def generate_recommendations(
    reading: dict,
    irrigation_forecast: float,
    nitrogen_forecast: float,
) -> list[Recommendation]:
    recs: list[Recommendation] = []

    rain_current = reading.get("chuva_mm", 0)
    delta_rain   = irrigation_forecast - rain_current
    if delta_rain > 20:
        recs.append(Recommendation(
            priority=Priority.HIGH,
            parameter="Irrigacao",
            current_value=round(rain_current, 1),
            optimal_range=f">= {irrigation_forecast:.0f} mm",
            action=(
                f"Modelo preve necessidade de {irrigation_forecast:.0f} mm de agua. "
                f"Deficit estimado: {delta_rain:.0f} mm. Programar irrigacao."
            ),
            impact_pct=min(round(delta_rain / irrigation_forecast * 20, 1), 30),
        ))
    elif delta_rain < -30:
        recs.append(Recommendation(
            priority=Priority.MEDIUM,
            parameter="Irrigacao",
            current_value=round(rain_current, 1),
            optimal_range=f"aprox. {irrigation_forecast:.0f} mm",
            action=(
                f"Excesso de agua estimado ({abs(delta_rain):.0f} mm acima do necessario). "
                "Verificar drenagem ou reduzir irrigacao."
            ),
            impact_pct=5.0,
        ))

    n_current = reading.get("nitrogenio", 0)
    delta_n   = nitrogen_forecast - n_current
    if delta_n > 15:
        recs.append(Recommendation(
            priority=Priority.CRITICAL if delta_n > 40 else Priority.HIGH,
            parameter="Nitrogenio (N)",
            current_value=round(n_current, 1),
            optimal_range=f">= {nitrogen_forecast:.0f} mg/kg",
            action=(
                f"Deficit de N estimado em {delta_n:.0f} mg/kg. "
                "Aplicar ureia ou nitrato de amonio na proxima adubacao."
            ),
            impact_pct=min(round(delta_n / nitrogen_forecast * 25, 1), 35),
        ))

    ph = reading.get("ph", 6.5)
    ph_min, ph_max = OPTIMAL_RANGES["ph"]
    if ph < ph_min:
        recs.append(Recommendation(
            priority=Priority.HIGH,
            parameter="pH do Solo",
            current_value=round(ph, 2),
            optimal_range=f"{ph_min}–{ph_max}",
            action=f"Solo acido (pH={ph:.1f}). Aplicar calcario dolomitico para correcao.",
            impact_pct=min(round((ph_min - ph) / ph_min * 20, 1), 25),
        ))
    elif ph > ph_max:
        recs.append(Recommendation(
            priority=Priority.MEDIUM,
            parameter="pH do Solo",
            current_value=round(ph, 2),
            optimal_range=f"{ph_min}–{ph_max}",
            action=f"Solo alcalino (pH={ph:.1f}). Aplicar enxofre elementar.",
            impact_pct=10.0,
        ))

    humidity = reading.get("umidade", 60)
    h_min, h_max = OPTIMAL_RANGES["umidade"]
    if humidity < h_min:
        recs.append(Recommendation(
            priority=Priority.HIGH if humidity < 30 else Priority.MEDIUM,
            parameter="Umidade",
            current_value=round(humidity, 1),
            optimal_range=f"{h_min}–{h_max}%",
            action=f"Umidade baixa ({humidity:.0f}%). Aumentar frequencia de irrigacao.",
            impact_pct=10.0,
        ))

    if not recs:
        recs.append(Recommendation(
            priority=Priority.OK,
            parameter="Todas as variaveis",
            current_value=0,
            optimal_range="—",
            action="Condicoes dentro dos parametros ideais. Manter manejo atual.",
        ))

    recs.sort(key=lambda r: _SORT_ORDER.index(r.priority))
    return recs
