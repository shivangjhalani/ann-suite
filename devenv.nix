{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  # https://devenv.sh/basics/
  env.GREET = "Welcome to ann-suite development environment";
  # BCC is provided by Nix, not PyPI.  Expose its bindings before uv's virtual
  # environment so `from bcc import BPF` resolves to the real BCC module.
  env.PYTHONPATH = "${pkgs.python312Packages.bcc}/${pkgs.python312.sitePackages}";

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.docker
    pkgs.stdenv.cc.cc.lib # for many python wheels on linux
    pkgs.bcc
    pkgs.python312Packages.bcc
    pkgs.kmod # lets BCC load CONFIG_IKHEADERS when needed
    pkgs.glib
    pkgs.gcc
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.12";
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  # Fix for python wheels needing libstdc++.so.6 and other common libs
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.glib
  ];

  enterShell = ''
    echo $GREET
    python --version
    uv --version
    docker --version
    # Symlink for IDE compatibility
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
      ln -s "$DEVENV_STATE/venv" "$DEVENV_ROOT/.venv"
    fi
  '';
}
