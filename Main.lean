import LeanAX.StableHLO
import LeanAX.Validate

namespace LeanAX.Cli

def usage : String :=
  "leanax commands:\n" ++
  "  emit-stablehlo --case NAME --out PATH [--manifest-out PATH]\n" ++
  "  list-cases\n" ++
  "\n" ++
  "Available cases: " ++ LeanAX.joinSep ", " LeanAX.availableCases ++ "\n"

partial def parseFlag (flag : String) : List String -> Option String
  | [] => none
  | x :: y :: xs => if x == flag then some y else parseFlag flag (y :: xs)
  | _ :: xs => parseFlag flag xs

def emitStableHLO (args : List String) : IO UInt32 := do
  let caseName := parseFlag "--case" args
  let outPath := parseFlag "--out" args
  let manifestOutPath := parseFlag "--manifest-out" args
  match caseName, outPath with
  | some name, some path =>
      match LeanAX.moduleByName name with
      | some modu =>
          match modu.validate with
          | .ok () =>
              IO.FS.writeFile path modu.render
              if let some manifestPath := manifestOutPath then
                IO.FS.writeFile manifestPath (modu.renderLoweringManifest path)
                IO.println s!"wrote {manifestPath}"
              IO.println s!"wrote {path}"
              pure 0
          | .error err =>
              IO.eprintln s!"validation failed for case '{name}': {err.render}"
              pure 1
      | none =>
          IO.eprintln s!"unknown case '{name}'"
          IO.eprintln usage
          pure 2
  | _, _ =>
      IO.eprintln usage
      pure 2

def main (args : List String) : IO UInt32 := do
  match args with
  | [] =>
      IO.println usage
      pure 0
  | "emit-stablehlo" :: rest => emitStableHLO rest
  | ["list-cases"] =>
      IO.println (LeanAX.joinSep "\n" LeanAX.availableCases)
      pure 0
  | ["--help"] =>
      IO.println usage
      pure 0
  | ["-h"] =>
      IO.println usage
      pure 0
  | command :: _ =>
      IO.eprintln s!"unknown command '{command}'"
      IO.eprintln usage
      pure 2

end LeanAX.Cli

def main (args : List String) : IO UInt32 :=
  LeanAX.Cli.main args
