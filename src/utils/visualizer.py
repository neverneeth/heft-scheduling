"""
Visualization utilities for DAGs and scheduling results.

This module provides functions to visualize workflow DAGs, Gantt charts,
and other scheduling-related diagrams.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Optional, Dict, List
from src.core.workflow_dag import WorkflowDAG
from src.core.schedule_result import ScheduleResult


class Visualizer:
    """
    Provides visualization capabilities for DAGs and schedules.
    
    This class contains static methods for creating various visualizations
    including DAG structure diagrams and Gantt charts.
    """
    
    # Color palette for consistent visualization
    TASK_COLORS = [
        '#90C9E7', '#219EBC', '#136783', '#1597A5', '#FEB705',
        '#F3A261', '#FA8600', '#E9C46B', '#F4A261', '#2A9D8F',
        '#E76F51', '#F4A261', '#E9C46A', '#264653', '#287271'
    ]
    
    @staticmethod
    def visualize_dag(
        dag: WorkflowDAG,
        title: str = "Workflow DAG",
        figsize: tuple = (10, 6),
        show: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the structure of a workflow DAG.
        
        Args:
            dag: The workflow DAG to visualize
            title: Title for the plot
            figsize: Figure size (width, height)
            show: Whether to display the plot
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Create position layout based on topological layers
        pos = nx.multipartite_layout(dag.graph, subset_key="layer", align="horizontal")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw the graph
        nx.draw_networkx_nodes(
            dag.graph,
            pos,
            node_color='lightblue',
            node_size=800,
            ax=ax
        )
        
        nx.draw_networkx_labels(
            dag.graph,
            pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )
        
        nx.draw_networkx_edges(
            dag.graph,
            pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            ax=ax
        )
        
        # Add edge labels with communication costs
        edge_labels = {
            (u, v): f"{dag.get_communication_cost(u, v):.1f}"
            for u, v in dag.graph.edges()
        }
        nx.draw_networkx_edge_labels(
            dag.graph,
            pos,
            edge_labels,
            font_size=8,
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    @staticmethod
    def visualize_gantt_chart(
        result: ScheduleResult,
        title: Optional[str] = None,
        figsize: tuple = (12, 6),
        show: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a Gantt chart visualization of a schedule.
        
        Args:
            result: The scheduling result to visualize
            title: Title for the plot (defaults to algorithm name)
            figsize: Figure size (width, height)
            show: Whether to display the plot
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if title is None:
            title = f"Gantt Chart - {result.algorithm_name}"
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get number of processors
        num_processors = len(result.processor_schedules)
        
        # Draw task bars for each processor
        for proc_id, tasks in result.processor_schedules.items():
            for task_info in tasks:
                task = task_info['task']
                # Extract task number for color selection
                task_num = int(''.join(filter(str.isdigit, task)))
                color = Visualizer.TASK_COLORS[task_num % len(Visualizer.TASK_COLORS)]
                
                # Draw horizontal bar
                ax.barh(
                    y=proc_id,
                    width=task_info['duration'],
                    left=task_info['start'],
                    height=0.6,
                    color=color,
                    edgecolor='black',
                    linewidth=1
                )
                
                # Add task label
                ax.text(
                    task_info['start'] + task_info['duration'] / 2,
                    proc_id,
                    task,
                    ha='center',
                    va='center',
                    fontsize=10,
                    fontweight='bold'
                )
        
        # Configure axes
        ax.set_yticks(range(num_processors))
        ax.set_yticklabels([f'P{i}' for i in range(num_processors)])
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Processor', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add makespan line
        ax.axvline(
            x=result.makespan,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Makespan = {result.makespan:.2f}'
        )
        
        # Set axis limits
        ax.set_xlim(0, result.makespan * 1.1)
        ax.set_ylim(-0.5, num_processors - 0.5)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    @staticmethod
    def visualize_convergence(
        convergence_history: List[float],
        title: str = "Q-Learning Convergence",
        window_size: int = 100,
        figsize: tuple = (10, 5),
        show: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize Q-learning convergence history.
        
        Args:
            convergence_history: List of mean absolute Q-value changes per episode
            title: Title for the plot
            window_size: Size of moving average window
            figsize: Figure size (width, height)
            show: Whether to display the plot
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot raw convergence data
        ax.plot(
            convergence_history,
            color='grey',
            alpha=0.5,
            label='Raw',
            linewidth=0.5
        )
        
        # Calculate and plot moving average
        if len(convergence_history) >= window_size:
            convergence_array = np.array(convergence_history)
            moving_avg = np.convolve(
                convergence_array,
                np.ones(window_size) / window_size,
                mode='valid'
            )
            ax.plot(
                range(window_size - 1, len(convergence_history)),
                moving_avg,
                color='blue',
                label=f'Moving Average (window={window_size})',
                linewidth=2
            )
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Mean Absolute Q-value Change', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    @staticmethod
    def compare_algorithms(
        results: List[ScheduleResult],
        figsize: tuple = (10, 6),
        show: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comparison bar chart of multiple algorithm results.
        
        Args:
            results: List of ScheduleResult objects to compare
            figsize: Figure size (width, height)
            show: Whether to display the plot
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Extract data
        algorithm_names = [r.algorithm_name for r in results]
        makespans = [r.makespan for r in results]
        avg_utilizations = [r.get_average_utilization() for r in results]
        
        # Plot makespans
        bars1 = ax1.bar(algorithm_names, makespans, color='skyblue', edgecolor='black')
        ax1.set_ylabel('Makespan', fontsize=12)
        ax1.set_title('Makespan Comparison', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # Plot average utilizations
        bars2 = ax2.bar(algorithm_names, avg_utilizations, color='lightcoral', edgecolor='black')
        ax2.set_ylabel('Average Utilization (%)', fontsize=12)
        ax2.set_title('Resource Utilization Comparison', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
