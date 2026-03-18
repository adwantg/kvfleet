"""kvfleet CLI — command-line interface for fleet management."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from kvfleet import __version__

app = typer.Typer(
    name="kvfleet",
    help="🚀 KV-cache-aware intelligent routing for self-hosted LLM fleets",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Show kvfleet version."""
    console.print(f"kvfleet v{__version__}")


@app.command()
def validate(
    config: str = typer.Argument(..., help="Path to fleet YAML config file"),
) -> None:
    """Validate a fleet configuration file."""
    from kvfleet.config.loader import load_config

    try:
        fleet = load_config(config)
        console.print(Panel.fit(
            f"[green]✓ Valid configuration[/green]\n"
            f"Fleet: {fleet.fleet_name}\n"
            f"Models: {len(fleet.models)}\n"
            f"Strategy: {fleet.strategy.value}",
            title="Config Validation",
        ))
    except FileNotFoundError:
        console.print(f"[red]✗ Config file not found: {config}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Validation error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def fleet(
    config: str = typer.Argument(..., help="Path to fleet YAML config file"),
) -> None:
    """Show fleet status and registered models."""
    from kvfleet.config.loader import load_config

    fleet_config = load_config(config)
    table = Table(title=f"Fleet: {fleet_config.fleet_name}")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="magenta")
    table.add_column("Endpoint", style="green")
    table.add_column("Quality", justify="right")
    table.add_column("Cost/1K", justify="right")
    table.add_column("Latency P50", justify="right")
    table.add_column("Enabled", justify="center")

    for model in fleet_config.models:
        table.add_row(
            model.name,
            model.provider.value,
            model.endpoint,
            f"{model.quality_score:.2f}",
            f"${model.cost_per_1k_input_tokens:.4f}",
            f"{model.latency_p50_ms:.0f}ms",
            "✓" if model.enabled else "✗",
        )

    console.print(table)
    console.print(f"\nStrategy: [bold]{fleet_config.strategy.value}[/bold]")
    console.print(f"Cache affinity: {'[green]enabled[/green]' if fleet_config.cache_affinity.enabled else '[red]disabled[/red]'}")
    console.print(f"Policy engine: {'[green]enabled[/green]' if fleet_config.policy.enabled else '[red]disabled[/red]'}")
    console.print(f"Shadow traffic: {'[green]enabled[/green]' if fleet_config.shadow.enabled else '[red]disabled[/red]'}")


@app.command()
def simulate(
    config: str = typer.Argument(..., help="Path to fleet YAML config file"),
    prompt: str = typer.Option("Hello, what can you help me with?", "--prompt", "-p", help="Prompt to simulate"),
    data_class: str = typer.Option("internal", "--data-class", "-d", help="Data classification"),
) -> None:
    """Simulate a routing decision without calling any backend."""
    from kvfleet.config.loader import load_config
    from kvfleet.router.engine import Router

    fleet_config = load_config(config)
    router = Router(fleet_config)

    explanation = asyncio.run(router.simulate(prompt=prompt, data_class=data_class))

    console.print(Panel.fit(
        explanation.summary(),
        title="Route Simulation",
        border_style="blue",
    ))


@app.command()
def health(
    config: str = typer.Argument(..., help="Path to fleet YAML config file"),
) -> None:
    """Check health of all fleet endpoints."""
    from kvfleet.config.loader import load_config
    from kvfleet.router.engine import Router

    fleet_config = load_config(config)
    router = Router(fleet_config)

    results = asyncio.run(router.health_check_all())

    table = Table(title="Endpoint Health")
    table.add_column("Endpoint", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Latency", justify="right")

    for endpoint, data in results.items():
        status = "[green]✓ healthy[/green]" if data.get("healthy") else "[red]✗ unhealthy[/red]"
        latency = f"{data.get('latency_ms', 0):.1f}ms"
        table.add_row(endpoint, status, latency)

    console.print(table)
    asyncio.run(router.close())


@app.command()
def explain(
    config: str = typer.Argument(..., help="Path to fleet YAML config file"),
    prompt: str = typer.Option("Hello", "--prompt", "-p", help="Prompt to explain routing for"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Explain a routing decision in detail."""
    from kvfleet.config.loader import load_config
    from kvfleet.router.engine import Router

    fleet_config = load_config(config)
    router = Router(fleet_config)

    explanation = asyncio.run(router.simulate(prompt=prompt))

    if output_json:
        console.print_json(json.dumps(explanation.to_dict(), indent=2))
    else:
        console.print(Panel.fit(
            explanation.summary(),
            title="Route Explanation",
            border_style="blue",
        ))


@app.command()
def serve(
    config: str = typer.Argument(..., help="Path to fleet YAML config file"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    api_key: str = typer.Option("", "--api-key", "-k", help="API key for authentication"),
) -> None:
    """Start the OpenAI-compatible gateway server."""
    from kvfleet.config.loader import load_config
    from kvfleet.gateway.server import run_gateway
    from kvfleet.router.engine import Router

    fleet_config = load_config(config)
    router = Router(fleet_config)

    console.print(Panel.fit(
        f"Starting kvfleet gateway\n"
        f"Host: {host}:{port}\n"
        f"Fleet: {fleet_config.fleet_name}\n"
        f"Models: {len(fleet_config.models)}\n"
        f"Strategy: {fleet_config.strategy.value}",
        title="🚀 kvfleet Gateway",
        border_style="green",
    ))

    run_gateway(router, host=host, port=port, api_key=api_key)


@app.command()
def init(
    output: str = typer.Option("fleet.yaml", "--output", "-o", help="Output config file path"),
) -> None:
    """Generate a sample fleet configuration file."""
    from kvfleet.config.schema import FleetConfig, ModelConfig, ProviderType
    from kvfleet.config.loader import save_config

    sample = FleetConfig(
        fleet_name="my-fleet",
        models=[
            ModelConfig(
                name="llama-3-8b-local",
                endpoint="http://localhost:8000",
                provider=ProviderType.VLLM,
                model_id="meta-llama/Llama-3-8B-Instruct",
                quality_score=0.7,
                cost_per_1k_input_tokens=0.0,
                latency_p50_ms=200,
                tags={"domain": "general", "tier": "fast"},
            ),
            ModelConfig(
                name="llama-3-70b-local",
                endpoint="http://gpu-server:8000",
                provider=ProviderType.VLLM,
                model_id="meta-llama/Llama-3-70B-Instruct",
                quality_score=0.9,
                cost_per_1k_input_tokens=0.0,
                latency_p50_ms=800,
                tags={"domain": "general", "tier": "quality"},
            ),
            ModelConfig(
                name="gpt-4o-fallback",
                endpoint="https://api.openai.com",
                provider=ProviderType.OPENAI_COMPAT,
                model_id="gpt-4o",
                quality_score=0.95,
                cost_per_1k_input_tokens=0.005,
                latency_p50_ms=400,
                allowed_data_classes=["public"],
                tags={"domain": "general", "tier": "premium"},
            ),
        ],
    )

    save_config(sample, output)
    console.print(f"[green]✓ Sample config written to {output}[/green]")
    console.print(f"Edit this file to match your fleet, then run: [cyan]kvfleet fleet {output}[/cyan]")


def main() -> None:
    """Entry point for the kvfleet CLI."""
    app()


if __name__ == "__main__":
    main()
