# R with what functai for R and its checks use, for `nix shell --impure -f r/tools/env.nix`.
# lmcc and lm15 are not on CRAN: r/check installs them from their checkouts.
let
  pkgs = (builtins.getFlake "nixpkgs").legacyPackages.${builtins.currentSystem};
  R = pkgs.rWrapper.override {
    packages = with pkgs.rPackages; [
      # functai's imports
      rlang vctrs tibble cli generics withr jsonlite
      # lm15's imports
      curl openssl askpass
      # tests and the tidyverse functai pairs with
      testthat dplyr tidyr purrr roxygen2 httpuv
      # tidymodels, and the vignettes
      parsnip dials workflows yardstick rsample tune recipes textrecipes glmnet knitr rmarkdown ggplot2 tidymodels
    ];
  };
in pkgs.buildEnv { name = "functai-r"; paths = [ R pkgs.pandoc pkgs.gcc pkgs.gnumake pkgs.pkg-config pkgs.openssl.dev pkgs.curl.dev ]; }
