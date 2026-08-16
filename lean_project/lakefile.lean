import Lake
open Lake DSL

package lean_project

require mathlib from git
  "https://mirror.sjtu.edu.cn/git/lean4-packages/mathlib4/" @ "v4.29.1"

lean_lib LeanProject
