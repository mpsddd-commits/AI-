import matplotlib.pyplot as plt

def create_earth_composition_pie_chart():
    """
    Generates and saves a pie chart of the Earth's crustal composition.
    """
    # Data from Wikipedia/Web Search (major elements in Earth's crust by weight)
    labels = ['Oxygen', 'Silicon', 'Aluminum', 'Iron', 'Calcium', 'Sodium', 'Potassium', 'Magnesium', 'Other']
    sizes = [46.6, 27.7, 8.1, 5.0, 3.6, 2.8, 2.6, 2.1, 1.5]
    explode = (0.05, 0, 0, 0, 0, 0, 0, 0, 0)  # explode 1st slice (Oxygen)

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    plt.title("Elemental Composition of Earth's Crust by Mass")

    # Save it to a file
    chart_path = 'app/earth_composition_pie_chart.png'
    plt.savefig(chart_path)
    print(f"Pie chart saved to {chart_path}")

if __name__ == '__main__':
    create_earth_composition_pie_chart()
