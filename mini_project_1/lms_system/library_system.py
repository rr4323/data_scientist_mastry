import os
import time

# Initial data structures
library = {
    "Python Basics": {"author": "Test", "available": True},
    "Data Science": {"author": "Test1", "available": False}
}

credentials = {
    "admin": "admin123",
    "librarian": "lib123"
}

issued_books = {}  # Dictionary to store issued books and their borrowers

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_welcome():
    """Print welcome message."""
    print("\n" + "="*50)
    print("Welcome to Library Management System")
    print("="*50 + "\n")

def display_books():
    """Display all books with their details."""
    print("\nCurrent Library Inventory:")
    print("-"*50)
    print(f"{'Title':<30} {'Author':<20} {'Status':<10}")
    print("-"*50)
    for title, details in library.items():
        status = "Available" if details["available"] else "Issued"
        print(f"{title:<30} {details['author']:<20} {status:<10}")
    print("-"*50)

def add_book():
    """Add a new book to the library."""
    print("\nAdding New Book")
    print("-"*30)
    title = input("Enter book title: ").strip()
    
    if title in library:
        print("Error: Book already exists in the library!")
        return
    
    author = input("Enter author name: ").strip()
    library[title] = {"author": author, "available": True}
    print(f"\nSuccessfully added '{title}' to the library!")

def issue_book():
    """Issue a book to a borrower."""
    print("\nIssuing a Book")
    print("-"*30)
    title = input("Enter book title to issue: ").strip()
    
    if title not in library:
        print("Error: Book not found in the library!")
        return
    
    if not library[title]["available"]:
        print("Error: Book is already issued!")
        return
    
    borrower = input("Enter borrower name: ").strip()
    library[title]["available"] = False
    issued_books[title] = borrower
    print(f"\nSuccessfully issued '{title}' to {borrower}!")

def return_book():
    """Return a book to the library."""
    print("\nReturning a Book")
    print("-"*30)
    title = input("Enter book title to return: ").strip()
    
    if title not in library:
        print("Error: Book not found in the library!")
        return
    
    if library[title]["available"]:
        print("Error: Book is already in the library!")
        return
    
    library[title]["available"] = True
    borrower = issued_books.pop(title)
    print(f"\nSuccessfully returned '{title}' from {borrower}!")

def view_issued_books():
    """Display all currently issued books."""
    if not issued_books:
        print("\nNo books are currently issued.")
        return
    
    print("\nCurrently Issued Books:")
    print("-"*50)
    print(f"{'Title':<30} {'Borrower':<20}")
    print("-"*50)
    for title, borrower in issued_books.items():
        print(f"{title:<30} {borrower:<20}")
    print("-"*50)

def login():
    """Handle user login."""
    attempts = 0
    while attempts < 3:
        username = input("\nEnter username: ").strip()
        password = input("Enter password: ").strip()
        
        if username in credentials and credentials[username] == password:
            print(f"\nWelcome, {username}!")
            return True
        
        attempts += 1
        if attempts < 3:
            print(f"Invalid credentials. {3-attempts} attempts remaining.")
        else:
            print("\nMaximum login attempts reached. Exiting...")
            return False

def main_menu():
    """Display and handle the main menu."""
    while True:
        print("\nMain Menu:")
        print("1. View Books")
        print("2. Add a Book")
        print("3. Issue a Book")
        print("4. Return a Book")
        print("5. View Issued Books")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            display_books()
        elif choice == "2":
            add_book()
        elif choice == "3":
            issue_book()
        elif choice == "4":
            return_book()
        elif choice == "5":
            view_issued_books()
        elif choice == "6":
            print("\nThank you for using Library Management System!")
            break
        else:
            print("\nInvalid choice! Please try again.")
        
        input("\nPress Enter to continue...")
        clear_screen()

def main():
    """Main function to run the library system."""
    clear_screen()
    print_welcome()
    
    if login():
        clear_screen()
        main_menu()
    else:
        print("\nGoodbye!")

if __name__ == "__main__":
    main() 