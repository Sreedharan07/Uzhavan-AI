from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, FrameBreak, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

doc = BaseDocTemplate("test_multi_frame.pdf", pagesize=letter, leftMargin=45, rightMargin=45, topMargin=54, bottomMargin=72)

title_h = 160
body_h = 792 - 54 - 72 - title_h

# Page 1 Frames
f_top = Frame(45, 792 - 54 - title_h, 522, title_h, id='top', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
f_left1 = Frame(45, 72, 252, body_h, id='left1', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
f_right1 = Frame(315, 72, 252, body_h, id='right1', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)

# Later Page Frames
f_left2 = Frame(45, 72, 252, 666, id='left2', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)
f_right2 = Frame(315, 72, 252, 666, id='right2', topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0)

pt1 = PageTemplate(id='FirstPage', frames=[f_top, f_left1, f_right1])
pt2 = PageTemplate(id='LaterPages', frames=[f_left2, f_right2])

doc.addPageTemplates([pt1, pt2])

styles = getSampleStyleSheet()
story = [
    Paragraph("<b>Title of IEEE Paper</b>", styles['Title']),
    Paragraph("Author Name, Affiliation", styles['Normal']),
    FrameBreak(),
    Paragraph("<b><i>Abstract</i>—This is the abstract text.</b>", styles['Normal']),
    Paragraph("Text in left column of page 1.", styles['Normal']),
]

doc.build(story)
print("Built test_multi_frame.pdf successfully")
