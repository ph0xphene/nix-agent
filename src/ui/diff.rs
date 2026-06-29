//! Git-style colored diff rendering plus the interactive apply confirmation.

use console::Style;
use dialoguer::Confirm;
use dialoguer::theme::ColorfulTheme;
use similar::{ChangeTag, TextDiff};

use super::{frame_header, frame_rule};

/// Render a `git diff`-style view of the change from `old_content` to
/// `new_content`, then ask the operator whether to apply it.
///
/// Returns `true` only on an explicit `y`. Any non-interactive terminal or I/O
/// error resolves to `false` — a denied apply is always the safe default, so we
/// never activate a system change we could not confirm.
#[must_use]
pub fn render_and_confirm_plan(old_content: &str, new_content: &str) -> bool {
    render_diff(old_content, new_content);

    println!();
    // `ColorfulTheme` renders the leading `?` prefix and the `[y/N]` suffix, so
    // the visible line reads: `? Apply this configuration to your system? [y/N]`.
    Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Apply this configuration to your system?")
        .default(false)
        .interact()
        .unwrap_or(false)
}

/// Print the colored unified diff inside the house frame. Deleted lines are red,
/// added lines green, context dimmed — line-numbered like `git diff`.
pub fn render_diff(old_content: &str, new_content: &str) {
    let del = Style::new().red();
    let add = Style::new().green();
    let ctx = Style::new().dim();
    let meta = Style::new().magenta().bold();

    println!();
    println!("{}", frame_header("Nix-Agent Evaluation Plan"));

    let diff = TextDiff::from_lines(old_content, new_content);

    if old_content == new_content {
        println!("{}", ctx.apply_to("  (no changes — configuration is identical)"));
        println!("{}", frame_rule());
        return;
    }

    for group in diff.grouped_ops(3) {
        for op in group {
            for change in diff.iter_changes(&op) {
                let (sign, style) = match change.tag() {
                    ChangeTag::Delete => ("-", &del),
                    ChangeTag::Insert => ("+", &add),
                    ChangeTag::Equal => (" ", &ctx),
                };
                let old_ln = fmt_lineno(change.old_index());
                let new_ln = fmt_lineno(change.new_index());

                println!(
                    "{} {} {} {}",
                    meta.apply_to(format!("{old_ln:>4}")),
                    meta.apply_to(format!("{new_ln:>4}")),
                    style.clone().bold().apply_to(sign),
                    style.apply_to(change.value().trim_end_matches('\n')),
                );
            }
        }
        println!("{}", meta.apply_to("    ⋮"));
    }

    println!("{}", frame_rule());
}

/// Right-aligned line number, blank when the line does not exist on that side.
fn fmt_lineno(idx: Option<usize>) -> String {
    match idx {
        Some(i) => (i + 1).to_string(),
        None => String::new(),
    }
}
