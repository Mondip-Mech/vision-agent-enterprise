from __future__ import annotations

import bcrypt
import streamlit as st

import database as db


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def register(email: str, password: str, confirm: str) -> tuple[dict | None, str | None]:
    """Returns (user, error_message)."""
    email = email.strip().lower()
    if not email or "@" not in email or "." not in email:
        return None, "Enter a valid email address."
    if len(password) < 8:
        return None, "Password must be at least 8 characters."
    if password != confirm:
        return None, "Passwords do not match."
    if db.get_user_by_email(email):
        return None, "An account with that email already exists."
    try:
        user = db.create_user(email, hash_password(password))
        return user, None
    except Exception as exc:
        return None, f"Registration failed: {exc}"


def login(email: str, password: str) -> tuple[dict | None, str | None]:
    """Returns (user, error_message)."""
    user = db.get_user_by_email(email.strip().lower())
    if not user or not verify_password(password, user["password_hash"]):
        return None, "Invalid email or password."
    return user, None


def get_current_user() -> dict | None:
    return st.session_state.get("user")


def set_current_user(user: dict) -> None:
    st.session_state.user = user


def logout() -> None:
    for key in ["user", "active_brand", "results", "show_brand_form", "editing_brand_id"]:
        st.session_state.pop(key, None)
