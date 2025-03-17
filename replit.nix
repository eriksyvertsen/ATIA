{pkgs}: {
  deps = [
    pkgs.nano
    pkgs.rustc
    pkgs.pkg-config
    pkgs.openssl
    pkgs.libiconv
    pkgs.cargo
    pkgs.libxcrypt
  ];
}
