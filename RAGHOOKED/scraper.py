import requests
from bs4 import BeautifulSoup

# Function to scrape website and save content to .txt file
def scrape_and_save(url, output_file):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the relevant HTML elements containing the text content
        # Adjust these based on the structure of the website
        text_elements = soup.find_all('p')  # Example: find all <p> elements
        
        # Extract text from the HTML elements
        scraped_text = '\n'.join([element.get_text() for element in text_elements])
        
        # Write the scraped text to a .txt file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(scraped_text)
        
        print(f"Scraped content saved to {output_file}")
    else:
        print(f"Failed to scrape content from {url}")

# Example usage
url = 'https://example.com'  # Replace with the URL of the website you want to scrape
output_file = 'scraped_content.txt'  # Name of the output .txt file
scrape_and_save(url, output_file)
