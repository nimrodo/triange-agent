"""PROTOTYPE — throwaway. Prints mocked CLI output for the legal-QA agent
(wayfinder ticket: "Prototype CLI output format for cited, confidence-flagged
answers", issue #34) so the format can be reacted to before real code exists.

Run: uv run scripts/prototype_cli_output.py
"""

SEPARATOR = "-" * 60


def answered_example() -> str:
    return f"""
> מהו שיעור המס השולי המרבי על הכנסת יחיד?

{SEPARATOR}
לפי סעיף 121 לפקודה, שיעור המס השולי המרבי המוטל על הכנסתו
של יחיד עומד על 47%, בכפוף למדרגות המס הקבועות באותו סעיף.

מקורות:
  [1] סעיף 121(ב) — "מס הכנסה על הכנסתו של יחיד ... ל-47% מיתרת
      ההכנסה החייבת."
{SEPARATOR}
"""


def uncertain_example() -> str:
    return f"""
> האם ניתן לנכות הוצאות רכב פרטי לצורכי עבודה?

{SEPARATOR}
⚠ תשובה לא ודאית (התאמה חלקית, 0.61) — הפקודה מתייחסת לניכוי
הוצאות רכב באופן חלקי בלבד, וייתכן שהתשובה המלאה תלויה בתקנות
שאינן חלק ממסמך זה.

לפי המידע שאותר: ניתן לנכות הוצאות רכב בהתאם לתקרות ולתנאים
הקבועים בתקנות מכוח סעיף 32, אך הפקודה עצמה אינה מפרטת את
שיעורי הניכוי המדויקים.

מקורות (התאמה):
  [1] סעיף 32(1)(א)  (0.61) — "הוצאות שהוצאו לשם ייצור הכנסה
      ... יותרו בניכוי בהתאם לתקנות שיקבע שר האוצר."
{SEPARATOR}
"""


def not_found_example() -> str:
    return f"""
> מהו שיעור המע"מ הנוכחי?

{SEPARATOR}
✗ לא נמצאה התייחסות — פקודת מס הכנסה אינה עוסקת בשיעור המע"מ
(המוסדר בחוק מס ערך מוסף, חוק נפרד). לא ניתן להשיב על סמך
המסמך הנוכחי.

הסעיפים הקרובים ביותר שנבדקו, ולא עברו את סף ההתאמה (0.35):
  - סעיף 15  (0.22) — עוסק בניכוי מס במקור, לא במע"מ.
  - סעיף 3   (0.19) — הגדרת "הכנסה חייבת".
{SEPARATOR}
"""


VARIANTS = {
    "answered": answered_example,
    "uncertain": uncertain_example,
    "not_found": not_found_example,
}


if __name__ == "__main__":
    for name, fn in VARIANTS.items():
        print(f"=== {name} ===")
        print(fn())
