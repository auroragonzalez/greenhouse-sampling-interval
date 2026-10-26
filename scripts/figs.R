# =====================================================================
#  Paper B — figures (ggplot2)
#  Reads the CSVs in results/ written by two_campaigns.py, synthetic.py
#  and reconcile.py, and writes vector PDFs to figs/.
#
#    Rscript scripts/figs.R
#
#  Palette: first three categorical slots of the validated reference palette
#  (blue / orange / aqua), which pass all-pairs CVD checks; ordered quantities
#  (tau, washout share) use a single blue ramp, light -> dark. Identity is also
#  carried by shape or linetype so every figure survives grayscale print.
# =====================================================================
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(patchwork); library(scales)
})

here <- function(...) file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])), ...)
if (is.na(here())) here <- function(...) file.path(".", ...)
DATA <- here("..", "results"); OUT <- here("..", "figs"); dir.create(OUT, showWarnings = FALSE)

# ---- style ------------------------------------------------------------------
FONT <- "Latin Modern Roman"
BLUE <- "#2a78d6"; ORANGE <- "#eb6834"; AQUA <- "#1baf7a"
INK <- "#0b0b0b"; INK2 <- "#52514e"; GRID <- "#e6e5e1"; ZERO <- "#3a3a38"
SAFE <- "#eef4fb"
RAMP5 <- c("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
CAMP_COL <- c(August = BLUE, September = ORANGE)
CAMP_SHP <- c(August = 16, September = 17)

theme_paper <- function(base = 8.5) {
  theme_minimal(base_size = base, base_family = FONT) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(colour = GRID, linewidth = .3),
          axis.line = element_line(colour = INK2, linewidth = .3),
          axis.ticks = element_line(colour = INK2, linewidth = .3),
          axis.text = element_text(colour = INK2),
          axis.title = element_text(colour = INK),
          strip.text = element_text(colour = INK, face = "bold", hjust = 0),
          legend.title = element_text(colour = INK2),
          legend.text = element_text(colour = INK),
          legend.key.height = unit(9, "pt"),
          plot.margin = margin(4, 6, 4, 4))
}
theme_set(theme_paper())
save <- function(p, name, w, h) {
  ggsave(file.path(OUT, name), p, width = w, height = h, units = "in",
         device = cairo_pdf)
  cat("  ", name, "\n")
}
have <- function(f) file.exists(file.path(DATA, f))
zero_line <- geom_hline(yintercept = 0, colour = ZERO, linewidth = .45)
TAU <- 16.1

# ---- Fig 1: the real sweep, both campaigns ---------------------------------
real_file <- if (have("real_sweep.csv")) "real_sweep.csv" else "real_sweep_point.csv"
if (have(real_file)) {
  R <- read_csv(file.path(DATA, real_file), show_col_types = FALSE) |>
    filter(exterior == "station", learner == "hgb") |>
    mutate(Campaign = recode(campaign, august = "August", september = "September"),
           group = recode(group, VENT_ROOF = "Roof vents", VENT_SIDE = "Side vents"))
  if (!"lo" %in% names(R)) { R$lo <- NA_real_; R$hi <- NA_real_ }
  # the criterion's own boundaries: dt/tau = 4 (common) and dt/hold = 0.5
  bounds <- tibble(Campaign = c("August", "September"), x = c(60, 15))
  p <- ggplot(R, aes(dt_min, effect, colour = Campaign, shape = Campaign)) +
    annotate("rect", xmin = -Inf, xmax = 4 * TAU, ymin = -Inf, ymax = Inf,
             fill = SAFE) +
    geom_vline(data = bounds, aes(xintercept = x, colour = Campaign),
               linetype = "22", linewidth = .4, show.legend = FALSE) +
    zero_line +
    geom_linerange(aes(ymin = lo, ymax = hi), linewidth = .5,
                   position = position_dodge(width = 4), na.rm = TRUE) +
    geom_line(linewidth = .6, position = position_dodge(width = 4)) +
    geom_point(size = 1.9, position = position_dodge(width = 4)) +
    facet_wrap(~group, nrow = 1) +
    scale_colour_manual(values = CAMP_COL, name = NULL) +
    scale_shape_manual(values = CAMP_SHP, name = NULL) +
    scale_x_continuous(breaks = c(10, 30, 45, 60, 90, 120)) +
    labs(x = "sampling interval Δt (min)",
         y = "do() effect of opening fully (°C)") +
    theme(legend.position = "top", legend.justification = "left",
          legend.margin = margin(0, 0, -4, 0))
  save(p, "fig1_real_sweep.pdf", 6.2, 2.7)
}

# ---- Fig 2: the true one-step sensitivity grows with the interval ----------
d2 <- expand_grid(tau = c(8, 16, 39), dt = seq(0, 120, by = 1)) |>
  mutate(g = -3 * (1 - exp(-dt / tau)),
         lab = factor(paste0("τ = ", tau, " min"), levels = paste0("τ = ", c(8, 16, 39), " min")))
p <- ggplot(d2, aes(dt, g, colour = lab, linetype = lab)) +
  geom_hline(yintercept = -3, colour = INK2, linewidth = .35, linetype = "13") +
  annotate("text", x = 118, y = -2.86, label = "static gain K", hjust = 1,
           family = FONT, size = 2.7, colour = INK2) +
  geom_line(linewidth = .7) +
  scale_colour_manual(values = RAMP5[c(1, 3, 5)], name = NULL) +
  scale_linetype_manual(values = c("solid", "42", "12"), name = NULL) +
  labs(x = "Δt (min)", y = expression("true " * partialdiff * T[t + Delta] / partialdiff * u ~ "(°C)")) +
  theme(legend.position = c(.98, .98), legend.justification = c(1, 1))
save(p, "fig2_analytic.pdf", 3.3, 2.4)

# ---- Fig 3: the three-condition factorial ----------------------------------
if (have("synth_factorial.csv")) {
  F <- read_csv(file.path(DATA, "synth_factorial.csv"), show_col_types = FALSE) |>
    filter(aprendiz == "hgb") |>
    mutate(lab = sprintf("washout %d min · %d input%s · Δt %d min",
                         as.integer(lavado_min), as.integer(n_entradas),
                         ifelse(n_entradas > 1, "s", ""), as.integer(dt_min)),
           campaign = lavado_min == 60 & n_entradas == 7 & dt_min == 60,
           lab = reorder(lab, ratio))
  p <- ggplot(F, aes(ratio, lab, fill = campaign)) +
    geom_vline(xintercept = 1, colour = INK2, linetype = "13", linewidth = .35) +
    geom_col(width = .62) +
    geom_vline(xintercept = 0, colour = ZERO, linewidth = .45) +
    annotate("text", x = 1.02, y = 8.35, label = "truth", hjust = 0,
             family = FONT, size = 2.7, colour = INK2) +
    scale_fill_manual(values = c(`FALSE` = BLUE, `TRUE` = ORANGE), guide = "none") +
    labs(x = "recovered / true gradient", y = NULL)
  save(p, "fig3_factorial.pdf", 5.2, 2.4)
}

# ---- Fig 4: collapse on dt/tau ---------------------------------------------
if (have("synth_taus.csv")) {
  T4 <- read_csv(file.path(DATA, "synth_taus.csv"), show_col_types = FALSE) |>
    mutate(tau = factor(paste0("τ = ", tau_min), levels = paste0("τ = ", sort(unique(tau_min)))))
  base4 <- list(zero_line, geom_line(linewidth = .55), geom_point(size = 1.3),
                scale_colour_manual(values = RAMP5, name = NULL))
  a <- ggplot(T4, aes(dt_min, ratio, colour = tau)) + base4 +
    labs(x = "Δt (min)", y = "recovered / true", subtitle = "against the raw interval")
  b <- ggplot(T4, aes(dt_over_tau, ratio, colour = tau)) + base4 +
    geom_vline(xintercept = 4, colour = INK2, linetype = "22", linewidth = .4) +
    scale_x_log10() +
    labs(x = "Δt / τ (log scale)", y = NULL, subtitle = "against Δt / τ")
  p <- (a | b) + plot_layout(guides = "collect") &
    theme(legend.position = "right", plot.subtitle = element_text(size = 8, colour = INK2))
  save(p, "fig4_collapse.pdf", 6.2, 2.4)
}

# ---- Fig 5: treatment length at fixed dt/tau -------------------------------
if (have("synth_treatment.csv")) {
  T5 <- read_csv(file.path(DATA, "synth_treatment.csv"), show_col_types = FALSE)
  p <- ggplot(T5, aes(dt_over_treat, ratio)) +
    annotate("rect", xmin = .2, xmax = .5, ymin = -Inf, ymax = Inf, fill = SAFE) +
    zero_line +
    geom_linerange(aes(ymin = ratio - ratio_sd, ymax = ratio + ratio_sd),
                   colour = BLUE, linewidth = .5) +
    geom_line(colour = BLUE, linewidth = .6) + geom_point(colour = BLUE, size = 1.8) +
    scale_x_log10(breaks = c(.25, .5, 1, 2, 4)) +
    labs(x = "Δt / treatment length (log scale)", y = "recovered / true")
  save(p, "fig5_treatment.pdf", 3.3, 2.4)
}

# ---- Fig 6: washout share, September geometry ------------------------------
if (have("synth_share.csv")) {
  S <- read_csv(file.path(DATA, "synth_share.csv"), show_col_types = FALSE) |>
    group_by(share, dt_min) |>
    summarise(m = mean(ratio), s = sd(ratio), .groups = "drop") |>
    mutate(lab = factor(ifelse(abs(share - 59 / 153) < 1e-6,
                               sprintf("%.2f (September)", share), sprintf("%.2f", share))),
           lab = factor(lab, levels = unique(lab[order(share)])))
  p <- ggplot(S, aes(dt_min, m, colour = lab)) +
    zero_line +
    geom_linerange(aes(ymin = m - s, ymax = m + s), linewidth = .45,
                   position = position_dodge(width = 5)) +
    geom_line(linewidth = .6, position = position_dodge(width = 5)) +
    geom_point(size = 1.5, position = position_dodge(width = 5)) +
    scale_colour_manual(values = RAMP5, name = "share of blocks\nending in washout") +
    scale_x_continuous(breaks = c(15, 30, 45, 60, 90, 120)) +
    labs(x = "Δt (min)", y = "recovered / true")
  save(p, "fig6_share.pdf", 5.2, 2.5)
}

# ---- Fig 7: three routes to the same gain ----------------------------------
if (have("reconcile.csv")) {
  E <- read_csv(file.path(DATA, "reconcile.csv"), show_col_types = FALSE) |>
    filter(grupo %in% c("VENT_ROOF", "VENT_SIDE", "FOG", "SHADE", "LIGHTS")) |>
    mutate(route = factor(recode(ruta, blockmean = "block mean (2 h)",
                                 onestep = "one-step (15 min)",
                                 step = "step test (asymptotic)"),
                          levels = c("block mean (2 h)", "one-step (15 min)", "step test (asymptotic)")),
           grupo = factor(recode(grupo, VENT_ROOF = "Roof vents", VENT_SIDE = "Side vents",
                                 FOG = "Fog", SHADE = "Shade", LIGHTS = "Lamps"),
                          levels = c("Roof vents", "Side vents", "Fog", "Shade", "Lamps")))
  p <- ggplot(E, aes(grupo, K_equivalente, fill = route)) +
    geom_col(position = position_dodge(width = .8), width = .72,
             colour = "white", linewidth = .5) +
    zero_line +
    scale_fill_manual(values = c(BLUE, ORANGE, AQUA), name = NULL) +
    labs(x = NULL, y = "equivalent static gain (°C)") +
    theme(legend.position = "top", legend.justification = "left",
          panel.grid.major.x = element_blank())
  save(p, "fig7_reconcile.pdf", 5.2, 2.6)
}

# ---- Fig 8: vent effect by interior-exterior temperature difference --------
if (have("vent_by_band.csv")) {
  V <- read_csv(file.path(DATA, "vent_by_band.csv"), show_col_types = FALSE) |>
    mutate(mid = (pmax(lo, -3) + pmin(hi, 18)) / 2,
           Campaign = recode(campaign, august = "August", september = "September"),
           group = recode(group, VENT_ROOF = "Roof vents", VENT_SIDE = "Side vents"),
           Interval = factor(paste(dt_min, "min"), levels = c("15 min", "60 min")))
  p <- ggplot(V, aes(mid, effect, colour = Interval, shape = Campaign,
                     linetype = Campaign)) +
    annotate("rect", xmin = -Inf, xmax = 0, ymin = -Inf, ymax = Inf, fill = SAFE) +
    zero_line +
    geom_line(linewidth = .55) +
    geom_point(aes(size = n)) +
    facet_wrap(~group, nrow = 1) +
    scale_colour_manual(values = c(`15 min` = BLUE, `60 min` = ORANGE), name = "Δt") +
    scale_shape_manual(values = CAMP_SHP, name = NULL) +
    scale_linetype_manual(values = c(August = "solid", September = "42"), name = NULL) +
    scale_size_area(max_size = 2.6, guide = "none") +
    labs(x = "interior − exterior temperature (°C, band midpoint)",
         y = "do() effect of opening fully (°C)") +
    theme(legend.position = "top", legend.justification = "left",
          legend.margin = margin(0, 0, -4, 0))
  save(p, "fig8_by_band.pdf", 6.2, 2.7)
}
cat("figures written to", OUT, "\n")
