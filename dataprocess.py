def normalize_user_text(s: str) -> str:
    return s.replace("<image> Text:", "<image>\nText:")


def strip_image_token_and_prefix(s: str) -> str:
    s = normalize_user_text(s)
    return s.replace("<image>", "").strip()