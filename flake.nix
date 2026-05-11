{
  description = "Lean-native array programming experiments targeting StableHLO";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f (import nixpkgs { inherit system; }));
    in
    {
      checks = forAllSystems (pkgs: {
        e2e = pkgs.runCommand "leanax-e2e"
          {
            nativeBuildInputs = [
              pkgs.lean4
              pkgs.cargo
              pkgs.rustc
              pkgs.stdenv.cc
              pkgs.llvmPackages.mlir
              pkgs.uv
              pkgs.python3
              pkgs.git
            ];
            src = self;
          } ''
          cp -R "$src" source
          chmod -R u+w source
          cd source
          lake build
          cargo test --locked --manifest-path e2e/runner/Cargo.toml
          cargo run --locked --manifest-path e2e/runner/Cargo.toml
          touch "$out"
        '';
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.lean4
            pkgs.cargo
            pkgs.rustc
            pkgs.rustfmt
            pkgs.llvmPackages.mlir
            pkgs.uv
            pkgs.python3
            pkgs.git
          ];
        };
      });

      formatter = forAllSystems (pkgs: pkgs.nixpkgs-fmt);
    };
}
