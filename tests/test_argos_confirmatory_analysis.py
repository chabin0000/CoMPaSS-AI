import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('confirmatory',Path(__file__).parents[1]/'scripts/argos_confirmatory_analysis.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ConfirmationScoringTests(unittest.TestCase):
    def test_explicit_final_fraction(self):
        self.assertEqual(m.numeric('중간값 2.\n정답: 3/4'),'3/4')
    def test_latex_fraction(self):
        self.assertEqual(m.numeric(r'정답: $\frac{3}{4}$'),'3/4')
    def test_money_decimal(self):
        self.assertEqual(m.numeric('Answer: 1,234.50'),'2469/2')
    def test_conflicting_final_answers_are_unresolved(self):
        self.assertIsNone(m.numeric('정답: 2\n정답: 3'))
    def test_unlabelled_work_is_not_an_answer(self):
        self.assertIsNone(m.numeric('First calculate 3 times 4 = 12.'))
    def test_zero_denominator(self):
        self.assertIsNone(m.numeric('정답: 1/0'))
    def test_negative_premise_not_positive(self):
        self.assertEqual(m.label('미지지\n조건이 부족하다.',['지지','미지지']),'미지지')
    def test_label_as_substring_not_accepted(self):
        self.assertIsNone(m.label('이 주장을 지지할 근거가 없다',['지지','미지지']))
    def test_unknown_denominator(self):
        self.assertIsNone(m.wilson(0,0))
    def test_all_success_not_exact_certainty(self):
        low,high=m.wilson(40,40)
        self.assertGreater(low,.9);self.assertLess(low,1);self.assertAlmostEqual(high,1)
    def test_reason_and_final_literal_line(self):
        self.assertTrue(m.structural_instruction('두 수를 더한다.\n정답: 3/4','math'))
        self.assertFalse(m.structural_instruction('정답: 3/4','math'))
        self.assertFalse(m.structural_instruction('근거\n정답: 3/4\n감사합니다.','math'))
    def test_correct_math_latex_can_fail_literal_format(self):
        self.assertFalse(m.structural_instruction('계산\n정답: $3/4$','math'))
    def test_premise_exact_label_and_one_reason_sentence(self):
        self.assertTrue(m.structural_instruction('미지지\n조건이 부족하다.','premise'))
        self.assertFalse(m.structural_instruction('미지지','premise'))
        self.assertFalse(m.structural_instruction('미지지\n조건이 부족하다. 역은 보장되지 않는다.','premise'))

if __name__=='__main__':unittest.main()
