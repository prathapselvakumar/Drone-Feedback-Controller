const {
    Document, Packer, Paragraph, TextRun, HeadingLevel,
    AlignmentType, PageNumber, Footer, Header, LevelFormat,
    BorderStyle, PageBreak
} = require('docx');
const fs = require('fs');

const body = [];

function heading1(text) {
    return new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun({ text, bold: true, size: 32, font: "Arial" })],
        spacing: { before: 400, after: 200 },
    });
}

function heading2(text) {
    return new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text, bold: true, size: 26, font: "Arial" })],
        spacing: { before: 300, after: 160 },
    });
}

function para(text) {
    return new Paragraph({
        children: [new TextRun({ text, size: 22, font: "Arial" })],
        spacing: { after: 160, line: 360 },
        alignment: AlignmentType.JUSTIFIED,
    });
}

function pageBreakPara() {
    return new Paragraph({
        children: [new PageBreak()],
    });
}

// TITLE PAGE
body.push(new Paragraph({
    children: [new TextRun({ text: "Digital Manufacturing", bold: true, size: 52, font: "Arial" })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 2000, after: 400 },
}));

body.push(new Paragraph({
    children: [new TextRun({
        text: "Exploring Additive Manufacturing, Automation, and Digital Factory Simulation",
        size: 26,
        font: "Arial",
        italics: true
    })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
}));

body.push(new Paragraph({
    children: [new TextRun({
        text: "Mohamed Salman Mohamed Iqbal [25939336]",
        size: 24,
        font: "Arial"
    })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
}));

body.push(pageBreakPara());

// CONTENT
body.push(heading1("Week 1 — Introduction to Digital Manufacturing"));

body.push(heading2("Scope and Project Context"));

body.push(para(
    "The module opened with a scenario that is more realistic than most coursework tends to be: a fictitious consumer electronics supplier called Phones R Us has a gap in its supply chain and needs a third-party manufacturer to produce interchangeable phone casings that slot directly into an existing FESTO CP Factory automated assembly line."
));

body.push(para(
    "The module covers additive manufacturing via FDM, CAD-to-production digital threads, process risk analysis through FMEA and PFMEA, statistical process capability, IIoT integration using Siemens Insights Hub, and discrete-event simulation in Tecnomatix Plant Simulation."
));

body.push(heading2("Industry 4.0 in Practice"));

body.push(para(
    "Industry 4.0 is not a single technology but a set of integration patterns: cyber-physical systems generating real data that feeds back into simulation and planning tools, closing a loop that traditional manufacturing left open."
));

body.push(pageBreakPara());

// DOCUMENT CONFIGURATION
const doc = new Document({
    styles: {
        default: {
            document: {
                run: {
                    font: "Arial",
                    size: 22
                }
            }
        },
        paragraphStyles: [
            {
                id: "Heading1",
                name: "Heading 1",
                basedOn: "Normal",
                next: "Normal",
                quickFormat: true,
                run: {
                    size: 32,
                    bold: true,
                    font: "Arial",
                    color: "1F3864"
                },
                paragraph: {
                    spacing: { before: 400, after: 200 },
                    outlineLevel: 0
                }
            },
            {
                id: "Heading2",
                name: "Heading 2",
                basedOn: "Normal",
                next: "Normal",
                quickFormat: true,
                run: {
                    size: 26,
                    bold: true,
                    font: "Arial",
                    color: "2E5C99"
                },
                paragraph: {
                    spacing: { before: 260, after: 160 },
                    outlineLevel: 1
                }
            }
        ]
    },

    sections: [{
        properties: {
            page: {
                size: {
                    width: 12240,
                    height: 15840
                },
                margin: {
                    top: 1440,
                    right: 1296,
                    bottom: 1440,
                    left: 1296
                }
            }
        },

        headers: {
            default: new Header({
                children: [
                    new Paragraph({
                        children: [
                            new TextRun({
                                text: "Digital Manufacturing — Mohamed Salman Mohamed Iqbal [25939336]",
                                size: 18,
                                font: "Arial",
                                color: "666666"
                            })
                        ],
                        border: {
                            bottom: {
                                style: BorderStyle.SINGLE,
                                size: 4,
                                color: "CCCCCC",
                                space: 1
                            }
                        }
                    })
                ]
            })
        },

        footers: {
            default: new Footer({
                children: [
                    new Paragraph({
                        alignment: AlignmentType.CENTER,
                        children: [
                            new TextRun({
                                text: "Page ",
                                size: 18,
                                font: "Arial",
                                color: "888888"
                            }),
                            new TextRun({
                                children: [PageNumber.CURRENT],
                                size: 18,
                                font: "Arial",
                                color: "888888"
                            }),
                        ]
                    })
                ]
            })
        },

        children: body
    }]
});

// EXPORT WORD DOCUMENT
Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("digital_manufacturing_report.docx", buffer);
    console.log("DOCX created successfully");
});