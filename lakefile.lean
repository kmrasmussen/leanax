import Lake
open Lake DSL

package «leanax» where

lean_lib LeanAX where

@[default_target]
lean_exe leanax where
  root := `Main
