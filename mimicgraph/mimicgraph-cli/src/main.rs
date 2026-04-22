use clap::Parser;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

mod artifacts;
mod cli;
pub mod eval;

pub use artifacts::WithMetadata;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry().with(fmt::layer()).init();
    let cli = cli::Cli::parse();
    cli.command.exec()
}
