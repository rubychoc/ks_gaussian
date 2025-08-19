



def truncate_after_second_user(messages, without_system_prompt=False):
    """
    Given a full text, will return only the turns:
    System
    User
    Assistant
    User
    """
    truncated = []
    assistant_count = 0

    for msg in messages:
        if msg["role"] == "system" and without_system_prompt:
            continue
        truncated.append(msg)
        if msg["role"] == "user":
            assistant_count += 1
        if assistant_count == 2:
            break

    return truncated