from nimbo.image_quality import CLIPImageChecker

class DummyChecker(CLIPImageChecker):
    def __init__(self, values):
        self.values = iter(values)
    def compare(self, image, descriptions):
        return [next(self.values), 0.0]

def test_quality_pass():
    report = DummyChecker([.9, .8]).evaluate(None, "sailboat", ["sailboat"])
    assert report.passed
    assert report.checks[0]["object"] == "sailboat"

def test_missing_object_escalates():
    report = DummyChecker([.85, .2]).evaluate(None, "sailboat", ["sailboat"])
    assert not report.passed
    assert report.reason == "low_alignment_or_missing_critical_object"

def test_no_critical_objects():
    report = DummyChecker([.9]).evaluate(None, "blue sky", [])
    assert report.passed
