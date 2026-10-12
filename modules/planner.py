
# ============================================================
# MODULE 7: AI Planning - Treatment Plan Generator
# Covers: Week 12 (AI Planning Techniques)
# ============================================================

from collections import deque
from typing import Dict, List, Set, Optional


class TreatmentPlanner:
    """
    STRIPS-based planner that generates action plans
    from an initial state to a goal state.
    """

    def __init__(self):
        self.action_library = self._build_action_library()

    def _build_action_library(self) -> List[Dict]:
        """Define the available planning actions."""

        return [
            {
                "name": "CallEmergencyServices",
                "precond": {"EMERGENCY_CASE", "PATIENT_PRESENT"},
                "delete": {"EMERGENCY_CASE"},
                "add": {"EMERGENCY_SERVICES_CALLED"},
                "cost": 0,
                "duration": "5 minutes",
            },
            {
                "name": "TransferToICU",
                "precond": {"EMERGENCY_SERVICES_CALLED", "ICU_AVAILABLE"},
                "delete": {"EMERGENCY_SERVICES_CALLED"},
                "add": {"PATIENT_IN_ICU", "MONITORING_ACTIVE"},
                "cost": 0,
                "duration": "15 minutes",
            },
            {
                "name": "OrderBloodPanel",
                "precond": {"PATIENT_PRESENT", "DIAGNOSIS_NEEDED"},
                "delete": {"DIAGNOSIS_NEEDED"},
                "add": {"BLOOD_RESULTS_PENDING"},
                "cost": 1,
                "duration": "30 minutes",
            },
            {
                "name": "ReceiveBloodResults",
                "precond": {"BLOOD_RESULTS_PENDING"},
                "delete": {"BLOOD_RESULTS_PENDING"},
                "add": {"BLOOD_RESULTS_AVAILABLE", "DIAGNOSIS_REFINED"},
                "cost": 0,
                "duration": "2 hours",
            },
            {
                "name": "OrderPCRTest",
                "precond": {"COVID_SUSPECTED", "PATIENT_PRESENT"},
                "delete": {"COVID_SUSPECTED"},
                "add": {"PCR_PENDING"},
                "cost": 1,
                "duration": "24 hours",
            },
            {
                "name": "ReceivePCRResult",
                "precond": {"PCR_PENDING"},
                "delete": {"PCR_PENDING"},
                "add": {"PCR_RESULT_AVAILABLE", "DIAGNOSIS_CONFIRMED"},
                "cost": 0,
                "duration": "24 hours",
            },
            {
                "name": "PrescribeAntiviral",
                "precond": {"DIAGNOSIS_CONFIRMED", "VIRAL_INFECTION"},
                "delete": {"VIRAL_INFECTION"},
                "add": {"ANTIVIRAL_PRESCRIBED", "TREATMENT_STARTED"},
                "cost": 1,
                "duration": "10 minutes",
            },
            {
                "name": "PrescribeAntibiotics",
                "precond": {"DIAGNOSIS_CONFIRMED", "BACTERIAL_INFECTION"},
                "delete": {"BACTERIAL_INFECTION"},
                "add": {"ANTIBIOTICS_PRESCRIBED", "TREATMENT_STARTED"},
                "cost": 1,
                "duration": "10 minutes",
            },
            {
                "name": "AdministerFluids",
                "precond": {"PATIENT_IN_ICU", "DEHYDRATION_RISK"},
                "delete": {"DEHYDRATION_RISK"},
                "add": {"FLUIDS_ADMINISTERED"},
                "cost": 1,
                "duration": "1 hour",
            },
            {
                "name": "MonitorVitals",
                "precond": {"PATIENT_PRESENT"},
                "delete": set(),
                "add": {"VITALS_MONITORED"},
                "cost": 0,
                "duration": "Continuous",
            },
            {
                "name": "IsolatePatient",
                "precond": {"CONTAGIOUS_DISEASE", "PATIENT_PRESENT"},
                "delete": {"CONTAGIOUS_DISEASE"},
                "add": {"PATIENT_ISOLATED"},
                "cost": 0,
                "duration": "As clinically directed",
            },
            {
                "name": "ScheduleFollowUp",
                "precond": {"TREATMENT_STARTED", "VITALS_MONITORED"},
                "delete": set(),
                "add": {"FOLLOWUP_SCHEDULED", "PLAN_COMPLETE"},
                "cost": 0,
                "duration": "5 minutes",
            },
        ]

    def _apply_action(
        self,
        state: frozenset,
        action: Dict,
    ) -> Optional[frozenset]:
        """Apply an action if all its preconditions are satisfied."""

        if not action["precond"].issubset(state):
            return None

        new_state = (state - action["delete"]) | action["add"]
        return frozenset(new_state)

    def generate_plan(
        self,
        initial_state: Set[str],
        goal_state: Set[str],
    ) -> Optional[List[Dict]]:
        """Find an action sequence using breadth-first search."""

        initial = frozenset(initial_state)
        goal = frozenset(goal_state)

        queue = deque([(initial, [])])
        visited = {initial}

        while queue:
            state, plan = queue.popleft()

            if goal.issubset(state):
                return plan

            for action in self.action_library:
                new_state = self._apply_action(state, action)

                if new_state is not None and new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, plan + [action]))

        return None

    def create_treatment_plan(
        self,
        diagnosis: str,
        urgency: str,
    ) -> Dict:
        """Generate a planning sequence for a diagnosis scenario."""

        diagnosis_key = diagnosis.lower().strip().replace(" ", "_")

        diagnosis_states = {
            "flu": {"VIRAL_INFECTION", "DIAGNOSIS_NEEDED"},
            "flu_suspected": {"VIRAL_INFECTION", "DIAGNOSIS_NEEDED"},
            "covid19": {
                "COVID_SUSPECTED",
                "CONTAGIOUS_DISEASE",
                "DIAGNOSIS_NEEDED",
            },
            "covid19_suspected": {
                "COVID_SUSPECTED",
                "CONTAGIOUS_DISEASE",
                "DIAGNOSIS_NEEDED",
            },
            "cardiac_event": {
                "EMERGENCY_CASE",
                "ICU_AVAILABLE",
            },
            "meningitis": {
                "EMERGENCY_CASE",
                "BACTERIAL_INFECTION",
                "ICU_AVAILABLE",
                "DIAGNOSIS_NEEDED",
            },
            "dengue": {
                "VIRAL_INFECTION",
                "DIAGNOSIS_NEEDED",
                "DEHYDRATION_RISK",
            },
            "tuberculosis": {
                "BACTERIAL_INFECTION",
                "CONTAGIOUS_DISEASE",
                "DIAGNOSIS_NEEDED",
            },
            "diabetes": {"DIAGNOSIS_NEEDED"},
            "common_cold": {
                "VIRAL_INFECTION",
                "DIAGNOSIS_NEEDED",
            },
        }

        initial_state = {"PATIENT_PRESENT"}
        initial_state.update(
            diagnosis_states.get(
                diagnosis_key,
                {"DIAGNOSIS_NEEDED"},
            )
        )

        # This prototype assumes a diagnosis is confirmed for
        # planning purposes; it does not confirm a real diagnosis.
        initial_state.add("DIAGNOSIS_CONFIRMED")

        goal_state = {
            "TREATMENT_STARTED",
            "VITALS_MONITORED",
            "FOLLOWUP_SCHEDULED",
        }

        if urgency.upper() in {"HIGH", "CRITICAL"}:
            initial_state.update({
                "EMERGENCY_CASE",
                "ICU_AVAILABLE",
            })
            goal_state.add("PATIENT_IN_ICU")

        plan = self.generate_plan(initial_state, goal_state)

        if plan is None:
            return {
                "diagnosis": diagnosis,
                "urgency": urgency,
                "error": "No valid plan found.",
                "steps": 0,
                "plan": [],
            }

        return {
            "diagnosis": diagnosis,
            "urgency": urgency,
            "initial_state": sorted(initial_state),
            "goal_state": sorted(goal_state),
            "steps": len(plan),
            "total_duration": (
                f"{len(plan)} actions; see individual durations"
            ),
            "plan": [
                {
                    "step": index + 1,
                    "action": action["name"],
                    "duration": action["duration"],
                    "cost": action["cost"],
                }
                for index, action in enumerate(plan)
            ],
            "summary": f"Planning sequence generated: {len(plan)} steps.",
        }

    def analyze(self, percept) -> Dict:
        """
        Agent interface.
        Uses a demonstration scenario until diagnosis and urgency
        are explicitly passed from the other modules.
        """

        result = self.create_treatment_plan(
            diagnosis="flu_suspected",
            urgency="MEDIUM",
        )

        result["summary"] = (
            "Demonstration planning sequence generated. "
            "Diagnosis and urgency are placeholders."
        )

        result["confidence"] = 0.7

        return result