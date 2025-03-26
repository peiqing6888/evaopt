"""
Retro-style demo script for EvaOpt with EVA-inspired colors
"""

import time
import os
from pyfiglet import Figlet
from termcolor import colored
import sys

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_slow(text, delay=0.03):
    """Print text slowly, character by character."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def print_ascii_art(text, font='slant', color='magenta'):
    """Print text in ASCII art style with EVA colors."""
    f = Figlet(font=font)
    ascii_art = f.renderText(text)
    print(colored(ascii_art, color))

def print_header():
    """Print the EvaOpt header in EVA style."""
    clear_screen()
    # Purple like EVA Unit-01
    print_ascii_art("EvaOpt", font='slant', color='magenta')
    # Blue like Rei's hair
    print_slow(colored("Welcome to EvaOpt - High Performance LLM Optimization Engine", 'blue'))
    # Green like NERV displays
    print("\n" + colored("="*70, 'green') + "\n")

def print_section(title, content, color='magenta'):
    """Print a section with EVA-themed colors."""
    print_ascii_art(title, font='small', color=color)
    print_slow(content)
    print()

def demo_optimization():
    """Run optimization demo with EVA-inspired visual effects."""
    print_header()
    
    # Show optimization methods
    methods = [
        ("SVD", "Singular Value Decomposition - Best for accuracy"),
        ("TruncatedSVD", "100x faster than full SVD"),
        ("LowRank", "Optimal for structured matrices")
    ]
    
    # Orange like EVA Unit-02
    print_section("Methods", "Available optimization techniques:", 'yellow')
    for method, desc in methods:
        # Blue like NERV interface text
        print_slow(f"• {colored(method, 'blue')}: {desc}")
    print("\n" + colored("-"*70, 'green') + "\n")
    
    # Show performance metrics
    print_section("Performance", "Key optimization results:", 'magenta')
    metrics = [
        ("Speed", "2x faster inference"),
        ("Memory", "60% reduction in memory usage"),
        ("Quality", "Maintained output quality (BLEU: 0.184)")
    ]
    
    for metric, value in metrics:
        # Orange like warning displays
        print_slow(f"• {colored(metric, 'yellow')}: {value}")
    print("\n" + colored("-"*70, 'green') + "\n")
    
    # Show optimization process
    print_section("Process", "Optimizing model...", 'blue')
    steps = [
        "Loading model configuration...",
        "Initializing optimization engine...",
        "Applying matrix compression...",
        "Quantizing weights...",
        "Optimization complete!"
    ]
    
    for step in steps:
        # Red like emergency alerts
        print_slow(f"[{colored('•', 'red')}] {step}")
        time.sleep(0.5)
    
    print("\n" + colored("="*70, 'green') + "\n")
    # Purple like EVA Unit-01 victory
    print_ascii_art("Done!", font='small', color='magenta')
    print_slow(colored("Thank you for using EvaOpt!", 'blue'))

if __name__ == "__main__":
    try:
        demo_optimization()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!") 