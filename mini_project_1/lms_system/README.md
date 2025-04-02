# Library Management System

A Python-based Library Management System that allows librarians to manage books, track issued books, and perform various library operations.

## Features

- User authentication with login system
- View all books in the library
- Add new books to the library
- Issue books to borrowers
- Return books to the library
- View all currently issued books
- Clean and user-friendly interface

## Prerequisites

- Python 3.x
- No additional dependencies required

## Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Run the program using Python:
   ```bash
   python library_system.py
   ```

## Usage

### Login Credentials
The system comes with two predefined user accounts:
- Username: `admin`, Password: `admin123`
- Username: `librarian`, Password: `lib123`

### Available Operations

1. **View Books**
   - Displays all books in the library with their title, author, and availability status

2. **Add a Book**
   - Add new books to the library
   - System checks for duplicate entries
   - Requires book title and author information

3. **Issue a Book**
   - Issue books to borrowers
   - Records borrower information
   - Updates book availability status

4. **Return a Book**
   - Process book returns
   - Updates book availability status
   - Removes borrower information

5. **View Issued Books**
   - Display all currently issued books
   - Shows borrower information for each issued book

6. **Exit**
   - Safely exit the program

## Security Features

- Login system with maximum 3 attempts
- Input validation for all operations
- Error handling for invalid operations

## Data Structure

The system uses the following data structures:
- Dictionary for storing book information
- Dictionary for user credentials
- Dictionary for tracking issued books

## Contributing

Feel free to submit issues and enhancement requests.