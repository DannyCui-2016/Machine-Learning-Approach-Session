previous_questions = {}

def detect_question_change(session_id, text):
    global previous_questions

    current_question = text.strip().split("\n")[0]

    if session_id not in previous_questions:
        previous_questions[session_id] = current_question
        return True, current_question

    if previous_questions[session_id] != current_question:
        previous_questions[session_id] = current_question
        return True, current_question

    return False, current_question
