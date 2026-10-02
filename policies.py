from .calibration import OnlineCalibrator,transform

class Policy:
    """Receives DecisionInput and only its own selected-pair Feedback."""
    def __init__(self,name,source):
        if name not in NAMES: raise ValueError("unknown normal policy")
        self.name=name; self.source=tuple(source)
        self.calibrator=OnlineCalibrator(source,offset=name=="rolling_offset") if name.startswith("rolling_") else None
    def predict_probabilities(self,decision):
        if self.name=="raw": return decision.raw.copy()
        if self.name=="source_platt": return transform(decision.raw,self.source)
        return self.calibrator.predict(decision.raw)
    def observe(self,feedback):
        if self.calibrator is not None: self.calibrator.observe(feedback)
    @property
    def params(self): return self.calibrator.params if self.calibrator else (self.source if self.name=="source_platt" else (1.,0.))

NAMES=("raw","source_platt","rolling_offset","rolling_platt")
