import unittest

from viral_factory.scoring import score_script_heuristics


class ScoringTests(unittest.TestCase):
    def test_price_hook_scores_higher_than_generic_copy(self) -> None:
        strong = score_script_heuristics(
            {
                "hook": "لو معاك 500 جنيه مين يقدر يعملها؟",
                "voiceover": "ابعت لينك الاكلة وحدد ميزانيتك وشوف مين في منطقتك هيبعتلك عرض.",
                "primary_caption": "مين في منطقتك يقدر يعملها؟",
                "cliffhanger": "تفتكر اول عميلة هتوافق ولا لا؟",
                "continuity_updates": ["كريم خسر شغله وقرر يبدأ من البيت"]
            }
        )
        weak = score_script_heuristics(
            {
                "hook": "اكتشف افضل تجربة طعام",
                "voiceover": "يمكنك من خلال منصتنا الوصول الى حلول مبتكرة للطعام.",
                "primary_caption": "انضم الان"
            }
        )
        self.assertGreater(strong["total"], weak["total"])


if __name__ == "__main__":
    unittest.main()
