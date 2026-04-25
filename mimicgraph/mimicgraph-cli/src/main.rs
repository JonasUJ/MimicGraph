use clap::Parser;
use clap_verbosity_flag::log;
use tracing_subscriber::Layer;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

mod artifacts;
mod cli;
pub mod eval;

pub use artifacts::WithMetadata;

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();

    let level = match cli.verbose.log_level_filter() {
        log::LevelFilter::Off => LevelFilter::OFF,
        log::LevelFilter::Error => LevelFilter::ERROR,
        log::LevelFilter::Warn => LevelFilter::WARN,
        log::LevelFilter::Info => LevelFilter::INFO,
        log::LevelFilter::Debug => LevelFilter::DEBUG,
        log::LevelFilter::Trace => LevelFilter::TRACE,
    };

    tracing_subscriber::registry()
        .with(fmt::layer().with_filter(level))
        .init();

    cli.command.exec()
}
