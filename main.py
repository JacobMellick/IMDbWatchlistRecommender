import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
import questionary

from naive_recommender import MovieRecommender
from ml_recommender import MLMovieRecommender

console = Console()


def display_results(df, title):
    """Uses Rich to print a pretty table of results"""
    console.print(f"\n[bold cyan]{title}[/bold cyan]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="white")
    table.add_column("Year", style="dim")
    table.add_column("Genre", style="cyan")
    table.add_column("Score/Pred", justify="right", style="green")

    # Handle different column names between the two scripts
    score_col = 'Predicted_Rating' if 'Predicted_Rating' in df.columns else 'Final_Score'

    for idx, row in df.iterrows():
        # Truncate genre if too long
        genre = row['Genres'] if len(
            row['Genres']) < 30 else row['Genres'][:27] + "..."
        table.add_row(
            str(idx + 1),
            row['Title'],
            str(row['Year']),
            genre,
            f"{row[score_col]:.2f}"
        )

    console.print(table)

    # Offer to show plot details
    if Confirm.ask("\nSee details for a specific movie?"):
        movie_idx = IntPrompt.ask("Enter the # number", choices=[
                                  str(i+1) for i in range(len(df))])
        selected = df.iloc[movie_idx-1]

        console.print(Panel(
            f"[bold]{selected['Title']} ({selected['Year']})[/bold]\n"
            f"[yellow]Genre:[/yellow] {selected['Genres']}\n"
            f"[yellow]Plot:[/yellow] {selected.get('Plot', 'No plot available')}\n"
            f"[yellow]Stars:[/yellow] {selected.get('Actors', 'N/A')}",
            title="Movie Details",
            border_style="green"
        ))


def filter_results(df):
    """Optional post-processing filters"""

    # 1. Runtime Filter
    if Confirm.ask("Do you have a time limit? (e.g. under 2 hours)"):
        minutes = IntPrompt.ask("Max runtime in minutes", default=120)
        if 'Runtime (mins)' in df.columns:
            df = df[df['Runtime (mins)'] <= minutes]

    # 2. Genre Filter
    if Confirm.ask("Are you in the mood for a specific genre?"):
        genre = Prompt.ask("Enter genre (e.g. Horror, Comedy)")
        df = df[df['Genres'].str.contains(genre, case=False, na=False)]

    return df


def main():
    console.clear()
    console.print(Panel.fit(
        "[bold yellow]🎬 AI Movie Recommender CLI[/bold yellow]", border_style="blue"))

    mode = questionary.select(
        "Choose your recommendation engine:",
        choices=[
            "1. Weighted Heuristic (Best for specific directors/genres you love)",
            "2. Gradient Boosting AI (Best for discovering hidden gems)",
            "3. Exit"
        ]
    ).ask()

    if "Exit" in mode:
        sys.exit()

    try:
        if "Weighted" in mode:
            console.print("[dim]Initializing Rule-Based Engine...[/dim]")
            engine = MovieRecommender()  # From naive_recommender
            engine.load_data()
            engine.analyze_user_preferences()

            with console.status("[bold green]Calculating scores & fetching details..."):
                recs = engine.get_recommendations(top_n=50)

        else:
            console.print("[dim]Initializing Machine Learning Engine...[/dim]")
            engine = MLMovieRecommender()  # From ml_recommender

            # Training Step
            console.print("[bold yellow]Step 1: Training Model[/bold yellow]")
            engine.train()

            # Prediction Step
            console.print(
                "\n[bold yellow]Step 2: Scoring Watchlist[/bold yellow]")
            recs = engine.predict()

        # --- Common Filtering & Display Logic ---
        console.print(f"\nGenerated [bold]{len(recs)}[/bold] candidates.")

        # Apply filters (from the previous main.py example)
        recs = filter_results(recs)

        if recs.empty:
            console.print("[red]No movies matched your filters![/red]")
        else:
            display_results(recs.head(15), "Top Recommendations")

    except Exception as e:
        console.print_exception()


if __name__ == "__main__":
    main()
