Create a Library Management System using Python. Below are the steps which you need to follow to create an end-to-end project for LMS. Here Librarian can add/modify/issue the books and can perform various other operation as listed in below steps. 
Use Python concepts only for creating this project.
Steps:
Step 1: Basic Structure
    1. Write a Python script that prints a welcome message for the Library Management System.
    2. Define an initial list or dictionary to store books, with details like title, author, and availability.
Example:
library = {
    "Python Basics": {"author": "Test", "available": True},
    "Data Science": {"author": "Test1", "available": False}
}
    3. Write a function to display all the books with their details.

Step 2: Adding Books
    4. Create a function to allow a user to add a new book to the system. The function should:
        ◦ Accept inputs for the title and author.
        ◦ Check if the book already exists in the dictionary.
        ◦ Add the book if it doesn’t exist.

Step 3: Issuing Books
    5. Create a function to issue a book. The function should:
        ◦ Check if the book exists and is available.
        ◦ Mark it as unavailable if issued and record the name of the person issuing it.
    6. Modify the display function to show the current status of the book (Available or Issued).

Step 4: Returning Books
    7. Create a function to return a book. The function should:
        ◦ Check if the book exists and is currently issued.
        ◦ Mark it as available again upon return.

Step 5: Login System
    8. Implement a login system:
        ◦ Store usernames and passwords in a dictionary. You can have some predefined stored credentials like below:
credentials = {"admin": "admin123", "librarian": "lib123"}
        ◦ Prompt users to log in before accessing the system.
        ◦ Allow up to 3 attempts before exiting.

Step 6: View Issued Books
    9. Add a function to view all books that are currently issued, along with the name of the person who issued them.

Step 7: Main Menu
    10. Combine all the functions into a menu-driven program:
        ◦ Use a while loop to display options like:
            ▪ View Books
            ▪ Add a Book
            ▪ Issue a Book
            ▪ Return a Book
            ▪ View Issued Books
            ▪ Exit