import Mathlib

example : (1 : ℕ) + 1 = 2 := by
  norm_num

example (x : ℕ) : x = x := by
  simp

example (x y : ℚ) (h : x ≤ y) : x - y ≤ 0 := by
  linarith
