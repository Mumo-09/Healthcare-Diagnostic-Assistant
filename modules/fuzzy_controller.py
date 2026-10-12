
# ============================================================
# MODULE 6: Fuzzy Logic - Patient Severity Assessment
# Covers: Week 12 (Fuzzy Logic)
# ============================================================


class FuzzySeverityAssessor:
    """
    Fuzzy logic system for patient severity assessment.

    Inputs: Temperature, Heart Rate, Symptom Count
    Output: Severity Score (0-100)
    """

    def _membership_temp(self, temp: float) -> dict:
        """Temperature membership functions."""
        return {
            "normal": max(0.0, min(1.0, (37.5 - temp) / 1.0))
            if temp <= 37.5 else 0.0,

            "mild": max(0.0, 1.0 - abs(temp - 38.0) / 1.0),

            "high": max(0.0, 1.0 - abs(temp - 39.0) / 1.0),

            "critical": max(0.0, min(1.0, (temp - 39.0) / 1.5))
            if temp >= 39.0 else 0.0
        }

    def _membership_hr(self, hr: int) -> dict:
        """Heart rate membership functions."""
        return {
            "low": max(0.0, min(1.0, (70 - hr) / 10.0))
            if hr <= 70 else 0.0,

            "normal": max(0.0, 1.0 - abs(hr - 80) / 20.0),

            "elevated": max(0.0, 1.0 - abs(hr - 100) / 15.0),

            "high": max(0.0, min(1.0, (hr - 100) / 20.0))
            if hr >= 100 else 0.0
        }

    def _membership_symptoms(self, count: int) -> dict:
        """Symptom count membership functions."""
        return {
            "few": max(0.0, min(1.0, (3 - count) / 2.0)),

            "moderate": max(0.0, 1.0 - abs(count - 4) / 2.0),

            "many": max(0.0, min(1.0, (count - 5) / 3.0))
        }

    def _defuzzify(self, severity_rules: dict) -> float:
        """Calculate the severity score using weighted averages."""
        centers = {
            "low": 15,
            "mild": 35,
            "moderate": 55,
            "high": 75,
            "critical": 92
        }

        numerator = sum(
            centers[label] * strength
            for label, strength in severity_rules.items()
        )

        denominator = sum(severity_rules.values())

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def assess(
        self,
        temperature: float,
        heart_rate: int,
        symptom_count: int
    ) -> dict:
        """Run the fuzzy inference process."""

        # Fuzzification
        temp_mf = self._membership_temp(temperature)
        hr_mf = self._membership_hr(heart_rate)
        symptom_mf = self._membership_symptoms(symptom_count)

        # Fuzzy rule evaluation
        rules = {
            "critical": max(
                min(temp_mf["critical"], hr_mf["high"]),
                min(temp_mf["critical"], symptom_mf["many"])
            ),

            "high": max(
                min(temp_mf["high"], hr_mf["elevated"]),
                min(temp_mf["high"], symptom_mf["many"]),
                min(temp_mf["mild"], hr_mf["high"])
            ),

            "moderate": max(
                min(temp_mf["mild"], hr_mf["normal"]),
                min(temp_mf["high"], symptom_mf["moderate"]),
                min(temp_mf["normal"], symptom_mf["many"])
            ),

            "mild": max(
                min(temp_mf["mild"], symptom_mf["few"]),
                min(temp_mf["normal"], symptom_mf["moderate"])
            ),

            "low": min(
                temp_mf["normal"],
                hr_mf["normal"],
                symptom_mf["few"]
            )
        }

        # Defuzzification and classification
        severity_score = self._defuzzify(rules)
        severity_label = self._classify(severity_score)

        return {
            "severity_score": round(severity_score, 2),
            "severity_label": severity_label,
            "rule_strengths": {
                label: round(strength, 3)
                for label, strength in rules.items()
            },
            "memberships": {
                "temperature": temp_mf,
                "heart_rate": hr_mf,
                "symptoms": symptom_mf
            }
        }

    def _classify(self, score: float) -> str:
        """Convert the score into a severity category."""
        if score >= 80:
            return "CRITICAL"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MODERATE"
        elif score >= 20:
            return "MILD"
        return "LOW"

    def analyze(self, percept) -> dict:
        """Interface used by the healthcare diagnostic agent."""

        result = self.assess(
            temperature=percept.temperature,
            heart_rate=percept.heart_rate,
            symptom_count=len(percept.symptoms)
        )

        result["summary"] = (
            f"Severity: {result['severity_label']} "
            f"({result['severity_score']:.1f}/100)"
        )

        result["diagnosis"] = result["severity_label"]
        result["confidence"] = result["severity_score"] / 100.0

        return result