import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# Constants
VISUALS_DIR = "visuals"
os.makedirs(VISUALS_DIR, exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the typing data"""
    df = pd.read_csv("typing_data.csv")
    
    # Data cleaning and transformations
    df["test_date"] = pd.to_datetime(df["test_date"])
    df["week"] = df["test_date"].dt.isocalendar().week
    df["day_of_week"] = df["test_date"].dt.day_name()
    df["session_duration_min"] = df["session_duration_ms"] / (1000 * 60)
    
    # Calculate error rate for additional insights
    df["error_rate"] = 100 - df["accuracy"]
    
    return df

def save_plot(fig, filename, width=1000, height=600):
    """Save plot with consistent sizing"""
    fig.update_layout(width=width, height=height)
    fig.write_html(f"{VISUALS_DIR}/{filename}")
    # Also save as static image
    fig.write_image(f"{VISUALS_DIR}/{filename.replace('.html', '.png')}")

def create_scatter_plot(df):
    """Enhanced scatter plot with regression lines"""
    fig = px.scatter(
        df,
        x="wpm",
        y="accuracy",
        color="user_id",
        title="<b>Typing Speed vs Accuracy</b><br><sup>Each point represents a test session</sup>",
        hover_data=["user_id", "test_date", "session_duration_min"],
        labels={
            "wpm": "Words Per Minute (WPM)",
            "accuracy": "Accuracy (%)",
            "user_id": "User",
            "session_duration_min": "Duration (min)"
        },
        trendline="lowess",
        marginal_x="histogram",
        marginal_y="box"
    )
    
    # Add annotations
    fig.add_annotation(
        x=0.95, y=0.05,
        xref="paper", yref="paper",
        text="Higher WPM often correlates with lower accuracy",
        showarrow=False,
        bgcolor="white"
    )
    
    fig.update_layout(
        hovermode="closest",
        plot_bgcolor="rgba(240,240,240,0.9)"
    )
    
    save_plot(fig, "interactive_wpm_accuracy.html")

def create_avg_wpm_chart(df):
    """Enhanced bar chart with performance comparison"""
    avg_stats = df.groupby("user_id").agg(
        avg_wpm=("wpm", "mean"),
        best_wpm=("wpm", "max"),
        consistency=("wpm", "std")
    ).reset_index()
    
    fig = px.bar(
        avg_stats,
        x="user_id",
        y="avg_wpm",
        color="avg_wpm",
        title="<b>Typing Performance Comparison</b>",
        labels={
            "user_id": "User ID",
            "avg_wpm": "Average WPM",
            "best_wpm": "Best WPM"
        },
        hover_data=["best_wpm", "consistency"],
        color_continuous_scale="Viridis"
    )
    
    # Add target line and annotations
    fig.add_hline(
        y=df["wpm"].mean(),
        line_dash="dot",
        annotation_text=f"Overall Average: {df['wpm'].mean():.1f} WPM",
        annotation_position="bottom right"
    )
    
    save_plot(fig, "bar_avg_wpm_by_user.html")

def create_progress_timeline(df):
    """Enhanced timeline with milestones"""
    fig = px.line(
        df.sort_values("test_date"),
        x="test_date",
        y="wpm",
        color="user_id",
        title="<b>Typing Speed Progress Over Time</b>",
        markers=True,
        line_shape="spline",
        hover_data=["accuracy", "session_duration_min"],
        labels={
            "wpm": "Words Per Minute",
            "test_date": "Date",
            "accuracy": "Accuracy (%)"
        }
    )
    
    # Add improvement rate calculation
    for user in df["user_id"].unique():
        user_data = df[df["user_id"] == user].sort_values("test_date")
        if len(user_data) > 1:
            improvement = ((user_data["wpm"].iloc[-1] - user_data["wpm"].iloc[0]) / 
                         user_data["wpm"].iloc[0] * 100)
            fig.add_annotation(
                x=user_data["test_date"].iloc[-1],
                y=user_data["wpm"].iloc[-1],
                text=f"+{improvement:.1f}%",
                showarrow=True,
                arrowhead=1
            )
    
    save_plot(fig, "typing_speed_over_time.html")

def create_advanced_distribution(df):
    """Enhanced distribution analysis"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("WPM Distribution", "Accuracy Distribution"))
    
    # WPM distribution with kernel density
    fig1 = px.histogram(
        df, x="wpm", nbins=20, marginal="rug",
        title="", color_discrete_sequence=["#636EFA"]
    )
    fig.add_trace(fig1.data[0], row=1, col=1)
    
    # Accuracy distribution with box plot
    fig2 = px.box(df, x="accuracy", points="all")
    fig.add_trace(fig2.data[0], row=1, col=2)
    
    fig.update_layout(
        title_text="<b>Performance Distribution Analysis</b>",
        showlegend=False
    )
    
    save_plot(fig, "performance_distributions.html")

def create_daily_patterns(df):
    """Weekly patterns visualization"""
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["test_date"].dt.day_name()
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=weekday_order, ordered=True)
    
    fig = px.box(
        df.sort_values("day_of_week"),
        x="day_of_week",
        y="wpm",
        color="day_of_week",
        title="<b>Typing Performance by Day of Week</b>",
        labels={
            "wpm": "Words Per Minute",
            "day_of_week": "Day of Week"
        }
    )
    
    save_plot(fig, "daily_performance_patterns.html")

def create_user_dashboard(df):
    """Interactive dashboard for individual users"""
    for user in df["user_id"].unique():
        user_df = df[df["user_id"] == user].sort_values("test_date")
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "xy"}, {"type": "polar"}],
                   [{"type": "xy"}, {"type": "xy"}]],
            subplot_titles=(
                f"Progress Over Time (User {user})",
                "Accuracy vs Speed",
                "Weekly Performance",
                "Session Duration Impact"
            )
        )
        
        # Progress line
        fig.add_trace(
            go.Scatter(
                x=user_df["test_date"],
                y=user_df["wpm"],
                mode="lines+markers",
                name="WPM"
            ),
            row=1, col=1
        )
        
        # Polar plot for accuracy vs speed
        fig.add_trace(
            go.Scatterpolar(
                r=user_df["accuracy"],
                theta=user_df["wpm"],
                mode="markers",
                name="Accuracy/WPM",
                marker=dict(
                    size=8,
                    color=user_df["session_duration_min"],
                    colorscale="Viridis",
                    showscale=True
                )
            ),
            row=1, col=2
        )
        
        # Weekly performance
        week_avg = user_df.groupby("week")["wpm"].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=week_avg["week"],
                y=week_avg["wpm"],
                name="Weekly Avg"
            ),
            row=2, col=1
        )
        
        # Duration impact
        fig.add_trace(
            go.Scatter(
                x=user_df["session_duration_min"],
                y=user_df["wpm"],
                mode="markers",
                name="Duration Impact",
                marker=dict(
                    size=user_df["accuracy"],
                    sizemode="area",
                    sizeref=2.*max(user_df["accuracy"])/(40.**2),
                    sizemin=4
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=f"<b>User {user} Typing Performance Dashboard</b>",
            height=800,
            showlegend=False
        )
        
        save_plot(fig, f"user_{user}_dashboard.html")

def main():
    print(f"🚀 Starting visualization generation at {datetime.now()}")
    
    df = load_and_preprocess_data()
    
    print("📊 Creating visualizations...")
    create_scatter_plot(df)
    create_avg_wpm_chart(df)
    create_progress_timeline(df)
    create_advanced_distribution(df)
    create_daily_patterns(df)
    create_user_dashboard(df)
    
    print(f"✅ All visualizations saved to '{VISUALS_DIR}' folder at {datetime.now()}")

if __name__ == "__main__":
    main()