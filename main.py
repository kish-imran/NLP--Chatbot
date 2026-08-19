from chatbot import get_response


# -------------------------------------------------
# Main Chatbot Program
# -------------------------------------------------

def main():

    print("=" * 60)
    print("🤖 NLP CHATBOT")
    print("=" * 60)

    print("\nType your message below.")

    print("Type 'bye' or 'exit' to close the chatbot.")

    print("=" * 60)


    while True:

        user_input = input("\nYou: ")


        # Exit commands
        if user_input.lower().strip() in [
            "bye",
            "goodbye",
            "exit",
            "quit"
        ]:

            print("Bot: Goodbye! Have a nice day!")

            break


        # Get chatbot response
        response = get_response(user_input)


        print("Bot:", response)


# -------------------------------------------------
# Run program
# -------------------------------------------------

if __name__ == "__main__":

    main()