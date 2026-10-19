"""Generate sample/acme_handbook.pdf: a fictional 6-page employee handbook with verifiable facts.

Run: python sample/make_sample.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUTPUT = Path(__file__).with_name("acme_handbook.pdf")

# One section per page: (title, [paragraphs])
SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Acme Corp Employee Handbook",
        [
            "Version 4.2, effective 1 March 2026. This handbook applies to all permanent and fixed-term "
            "employees of Acme Corp in every office. It replaces version 4.1 from January 2025.",
            "Acme Corp was founded in 2009 in Rotterdam and builds logistics software for mid-sized "
            "shipping companies. Today we have around 420 employees across offices in Rotterdam, "
            "Gothenburg and Lisbon.",
            "Our working week is 40 hours. Core hours, when everyone is expected to be reachable, are "
            "10:00 to 15:00 local time, Monday to Friday. Outside core hours you plan your own day "
            "together with your team.",
            "Questions about this handbook go to the People team at people@acme.example. Policy changes "
            "are announced in the #announcements channel at least 30 days before they take effect.",
        ],
    ),
    (
        "1. Onboarding",
        [
            "Every new employee starts on a Monday. Your manager sends a welcome email with your first-day "
            "schedule at least five working days before your start date.",
            "On day one you receive a laptop, a security key and access to the internal tools. Each new "
            "employee is paired with an onboarding buddy from another team for their first 90 days.",
            "Onboarding follows a 30-60-90 day plan. By day 30 you should have completed the mandatory "
            "training modules, by day 60 you should own a small project, and by day 90 you have a formal "
            "check-in with your manager.",
            "The probation period is six months. Either side may end employment during probation with "
            "two weeks' notice. After probation the standard notice period is one month.",
            "New employees receive a one-time home office budget of EUR 500 to buy a monitor, chair or "
            "other equipment. Receipts must be submitted within 60 days of the start date.",
        ],
    ),
    (
        "2. Vacation and Leave",
        [
            "Full-time employees receive 25 paid vacation days per calendar year. Part-time employees "
            "receive vacation days pro rata to their contracted hours. After five years of service you "
            "receive two additional vacation days per year.",
            "Up to five unused vacation days may be carried over to the next year. Carried-over days "
            "expire on 31 March of the following year if they have not been used.",
            "Vacation requests must be submitted in the HR portal at least two weeks in advance. Requests "
            "for more than ten consecutive working days need approval at least six weeks in advance.",
            "Sick leave: notify your manager before 09:30 on the first day of illness. A doctor's note is "
            "required from the fourth consecutive day of sickness.",
            "Parental leave: Acme Corp provides 16 weeks of fully paid parental leave for every new "
            "parent, which can be taken at any time within the first 12 months after birth or adoption.",
        ],
    ),
    (
        "3. Remote Work",
        [
            "Acme Corp follows a hybrid model. Employees may work remotely up to three days per week. "
            "Tuesdays are team days, when everyone is expected in the office.",
            "Fully remote work requires written approval from your manager and the People team. Working "
            "from another country is allowed for up to 20 working days per calendar year, provided the "
            "country is within the EU.",
            "Remote employees receive a monthly internet allowance of EUR 30, paid with the salary.",
            "When working remotely you must use the company VPN and must not work from public networks "
            "without it. Video calls should be taken from a private space where screens are not visible "
            "to others.",
        ],
    ),
    (
        "4. Expenses and Travel",
        [
            "Business expenses must be submitted in the Expensify app within 30 days of purchase, with a "
            "photo of the receipt attached. Expenses submitted later than 90 days are not reimbursed.",
            "Any single expense above EUR 250 requires pre-approval from your manager. Purchases above "
            "EUR 1,000 also require approval from the finance department.",
            "Travel: economy class is the standard for all flights. Business class is allowed only for "
            "flights longer than six hours. Trains are preferred for trips shorter than four hours.",
            "The hotel limit is EUR 180 per night in most cities and EUR 250 per night in London, Paris "
            "and New York. The daily meal allowance while travelling is EUR 60.",
            "Using a private car for business travel is reimbursed at EUR 0.23 per kilometre. Taxis are "
            "reimbursed only when public transport is not a reasonable option.",
        ],
    ),
    (
        "5. IT Security",
        [
            "Passwords must be at least 14 characters long. Use the company password manager, 1Password, "
            "for all work accounts and never reuse passwords between services.",
            "Multi-factor authentication with your hardware security key is mandatory for email, the VPN "
            "and all production systems.",
            "Laptops lock automatically after 5 minutes of inactivity and must use full-disk encryption. "
            "Security updates must be installed within 7 days of release.",
            "Report lost or stolen devices to the IT helpdesk within 1 hour, by calling +31 10 555 0199 "
            "or emailing security@acme.example. Suspected phishing emails are reported with the "
            "'Report phishing' button in Outlook.",
            "All employees complete security awareness training once a year. The deadline for the "
            "annual training is 30 November.",
        ],
    ),
]


def build(output: Path = OUTPUT) -> Path:
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paragraphs) in enumerate(SECTIONS):
        if i:
            story.append(PageBreak())
        story += [Paragraph(title, styles["Title" if i == 0 else "Heading1"]), Spacer(1, 0.4 * cm)]
        for text in paragraphs:
            story += [Paragraph(text, styles["BodyText"]), Spacer(1, 0.3 * cm)]

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        invariant=True,  # byte-identical output on every run, so the committed PDF only changes with its content
        title="Acme Corp Employee Handbook",
        author="Acme Corp (fictional)",
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )
    doc.build(story)
    return output


if __name__ == "__main__":
    print(f"Wrote {build()}")
