import argparse
import logging
import shutil
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from sapien.core.indexer import build_index
from sapien.core.limit_memory import start_memory_monitor
from sapien.core.merge_indexes import merge_in_batches
from sapien.core.searcher import Searcher
from sapien.core.tokenizer import Tokenizer


def main():
    console = Console()

    # --- Argument parser ---
    parser = argparse.ArgumentParser(
        description="Sapien Indexer CLI — build inverted indexes from a Wikipedia corpus "
        "into compressed blocks, with optional merging and search."
    )

    # Input / Output
    parser.add_argument("--input", required=True, help="Path to the input corpus file (.arrow)")
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where compressed index blocks (.zst) will be stored",
    )

    # Performance
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Documents per batch (default: auto — calculated based on available RAM)",
    )
    parser.add_argument("--min-length", type=int, default=3, help="Minimum token length")

    # Tokenizer
    parser.add_argument("--lowercase", action="store_true", help="Convert text to lowercase")
    parser.add_argument("--remove-punctuation", action="store_true", help="Remove punctuation")
    parser.add_argument("--remove-accents", action="store_true", help="Remove accents from words")
    parser.add_argument(
        "--remove-stopwords", action="store_true", help="Remove Portuguese stopwords (nltk)"
    )
    parser.add_argument(
        "--use-stemming", action="store_true", help="Enable stemming using Snowball (Portuguese)"
    )
    parser.add_argument("--keep-numbers", action="store_true", help="Keep numeric tokens")
    parser.add_argument(
        "--keep-alphanum", action="store_true", help="Keep alphanumeric tokens (e.g., 'a3b', 'x1')"
    )

    # System
    parser.add_argument(
        "--monitor-memory", action="store_true", help="Enable real-time memory usage monitoring"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG) logging")

    # Merge
    parser.add_argument(
        "--merge",
        action="store_true",
        help="After indexing, merge all index blocks into a single merged_index.zst file.",
    )

    # Search options
    parser.add_argument(
        "--search",
        type=str,
        help="Run a search query on the merged index after building (e.g., 'Portugal').",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to display for the search query.",
    )

    args = parser.parse_args()

    # --- Logging setup ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    # --- Memory monitor ---
    if args.monitor_memory:
        start_memory_monitor(show_memory_updates=True)

    # --- Start ---
    console.print("[bold green]Starting Sapien Indexer CLI...[/bold green]")

    data_path = Path(args.input)
    output_dir = Path(args.output)

    if not data_path.exists():
        console.print(f"[bold red]Input file not found:[/bold red] {data_path}")
        return

    # --- Cleanup old indexes ---
    if output_dir.exists():
        shutil.rmtree(output_dir)
        console.print(f"[yellow]Removed previous index directory:[/yellow] {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokenizer configuration ---
    tokenizer = Tokenizer(
        lowercase=args.lowercase,
        remove_punctuation=args.remove_punctuation,
        remove_accents_flag=args.remove_accents,
        min_length=args.min_length,
        remove_stopwords=args.remove_stopwords,
        use_stemming=args.use_stemming,
        keep_numbers=args.keep_numbers,
        keep_alphanum=args.keep_alphanum,
    )

    # Display config as a Rich table
    table = Table(title="[bold cyan]Tokenizer Configuration[/bold cyan]")
    table.add_column("Option", style="bold white", justify="right")
    table.add_column("Value", style="magenta")

    config_dict = {
        "lowercase": args.lowercase,
        "remove_punctuation": args.remove_punctuation,
        "remove_accents": args.remove_accents,
        "remove_stopwords": args.remove_stopwords,
        "use_stemming": args.use_stemming,
        "keep_numbers": args.keep_numbers,
        "keep_alphanum": args.keep_alphanum,
        "min_length": args.min_length,
        "batch_size": args.batch_size or "auto",
    }
    for k, v in config_dict.items():
        table.add_row(k, str(v))
    console.print(table)

    # --- Build index ---
    start_time = time.time()
    build_index(
        data_path=data_path, output_dir=output_dir, tokenizer=tokenizer, batch_size=args.batch_size
    )
    elapsed = time.time() - start_time
    console.print(f"[bold green]Index building complete in {elapsed:.2f}s.[/bold green]")

    # --- Merge (optional) ---
    if args.merge:
        console.print("\n[cyan]Merging index blocks into a single merged index...[/cyan]")

        # Define diretórios de origem e destino
        index_dir = output_dir
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)

        # Chama o merge hierárquico otimizado
        start_merge = time.time()
        merge_in_batches(index_dir, merged_dir, batch_size=3)
        merge_time = time.time() - start_merge

        final_path = merged_dir / "merged_index_final.zst"
        console.print(f"[bold green]Merge complete![/bold green] ({merge_time:.2f}s)")
        console.print(f"[yellow]Merged index saved to:[/yellow] {final_path}")

        merged_path = final_path

    # --- Search (optional) ---
    if args.search:
        if not merged_path.exists():
            console.print(f"[red]Error:[/red] merged index not found at {merged_path}")
            return

        console.print(f"\n[cyan]Running search for:[/cyan] [bold]{args.search}[/bold]")
        searcher = Searcher(index_path=merged_path, tokenizer=tokenizer)
        results = searcher.search(args.search, top_k=args.top_k)

        if results:
            console.print(f"[bold green]Top {len(results)} results:[/bold green]")
            for rank, (doc_id, score) in enumerate(results, start=1):
                title = searcher.doc_titles.get(str(doc_id), f"Documento {doc_id}")
                console.print(f"{rank:>2}. [bold]{title}[/bold] (ID={doc_id}) — score={score:.4f}")
        else:
            console.print(f"[yellow]No results found for query '{args.search}'[/yellow]")


if __name__ == "__main__":
    main()
