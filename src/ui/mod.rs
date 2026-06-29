//! Interactive presentation layer for the `nix-agent` CLI.
//!
//! Minimalist, high-contrast "netrunner" aesthetics built on `console`,
//! `dialoguer`, and `similar`. This is pure UI: it never touches the inference
//! core, the RAG index, or the activation engine — it only renders their output
//! and collects operator intent.

pub mod diff;

pub use diff::render_and_confirm_plan;

use console::Style;

/// Draw a full-width section header in the house style:
/// `── <title> ───────────────────────────────────────────────`.
///
/// Returns the rendered string so callers can print it however they like; the
/// helpers in this crate just `println!` it.
pub fn frame_header(title: &str) -> String {
    const WIDTH: usize = 72;
    let accent = Style::new().cyan().bold();
    // "── " + title + " " + trailing rule.
    let prefix = format!("── {title} ");
    let prefix_width = console::measure_text_width(&prefix);
    let fill = WIDTH.saturating_sub(prefix_width);
    let line = format!("{prefix}{}", "─".repeat(fill));
    accent.apply_to(line).to_string()
}

/// A plain full-width rule, matching [`frame_header`]'s width.
pub fn frame_rule() -> String {
    const WIDTH: usize = 72;
    Style::new().cyan().dim().apply_to("─".repeat(WIDTH)).to_string()
}
