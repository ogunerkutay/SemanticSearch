# Semantic Search Application

This C# application performs semantic search on text data using TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity. It's designed to search through a text file on the user's desktop and find the most semantically similar line to a given query.

## Features

- Reads text data from a file on the user's desktop
- Implements semantic search using TF-IDF and cosine similarity
- Uses Microsoft ML.NET for text feature extraction
- Returns the most relevant result based on semantic similarity

## Requirements

- .NET Core 3.1 or later
- Microsoft.ML NuGet package

## Usage

1. Place a text file named `news.txt` on your desktop. This file should contain the text data you want to search through.

2. Run the application. It will prompt you with a default search query: "Motorin satış fiyatı" (Diesel sale price).

3. The application will output the most semantically similar line from the `news.txt` file to the given query.

## Code Structure

- `Main`: Entry point of the application. Sets up the file path and initiates the search.
- `ReadLinesFromFile`: Reads all lines from the specified file.
- `SemanticSearch`: Implements the semantic search functionality using TF-IDF and cosine similarity.
- `CosineSimilarity`: Calculates the cosine similarity between two vectors.

## Classes

- `InputData`: Represents the input text data.
- `TransformedData`: Represents the text data after TF-IDF transformation.

## Future Improvements

- Add command-line arguments for custom file paths and queries.
- Implement multi-language support.
- Optimize for larger datasets.
- Add unit tests for core functionalities.

## License

[MIT License](https://opensource.org/licenses/MIT)

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/ogunerkutay/SemanticSearch/issues) if you want to contribute.
