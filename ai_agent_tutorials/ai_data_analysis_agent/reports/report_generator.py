import os
import time
from datetime import datetime
from pathlib import Path
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.pdfgen import canvas
import plotly.io as pio
import plotly.graph_objects as go
import cairosvg
import io
from PIL import Image as PILImage

# Jump brand colors
JUMP_COLORS = {
    'black': '#20204A',
    'dark_purple': '#32327A',
    'light_purple': '#D7BFED',
    'white': '#FFFFFF',
    'yellow': '#FFC958',
    'peach': '#FE7B94',
    'pink': '#EA7BCE',
    'blue': '#8EB7F8',
    'green': '#B4EC9D'
}

class ReportGenerator:
    def __init__(self):
        self.static_dir = Path(__file__).parent / 'static'
        self.temp_dir = self.static_dir / 'temp'
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='JumpTitle',
            parent=self.styles['Title'],
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=colors.HexColor(JUMP_COLORS['dark_purple']),
            spaceAfter=30,
            alignment=1  # Center alignment
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='JumpHeading',
            parent=self.styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            textColor=colors.HexColor(JUMP_COLORS['dark_purple']),
            spaceAfter=16
        ))
        
        # Normal text style
        self.styles.add(ParagraphStyle(
            name='JumpBody',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=12,
            textColor=colors.HexColor(JUMP_COLORS['black']),
            spaceAfter=12,
            leading=16  # Increased line height for better readability
        ))
        
        # Code style for SQL
        self.styles.add(ParagraphStyle(
            name='JumpCode',
            parent=self.styles['Code'],
            fontName='Courier',
            fontSize=10,
            textColor=colors.HexColor(JUMP_COLORS['dark_purple']),
            backColor=colors.HexColor('#F5F5F5'),
            spaceAfter=12,
            leading=14
        ))
        
    def _save_visualization(self, fig):
        """Save plotly figure as static image"""
        try:
            print("Debug: Attempting to save visualization")
            print(f"Debug: Figure type: {type(fig)}")
            
            timestamp = int(time.time())
            viz_path = self.temp_dir / f'visualization_{timestamp}.png'
            
            if fig is not None:
                print("Debug: Figure is not None, attempting to save...")
                # Create a copy of the figure for the report
                report_fig = fig.to_dict()
                fig_copy = go.Figure(report_fig)
                fig_copy.update_layout(
                    template='none',  # Use light theme
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color=JUMP_COLORS['dark_purple']),
                    title_font=dict(color=JUMP_COLORS['dark_purple']),
                    legend_font=dict(color=JUMP_COLORS['dark_purple'])
                )
                
                # Update axis colors
                fig_copy.update_xaxes(color=JUMP_COLORS['dark_purple'], gridcolor='#E5E5E5')
                fig_copy.update_yaxes(color=JUMP_COLORS['dark_purple'], gridcolor='#E5E5E5')
                
                # Save with high resolution
                pio.write_image(
                    fig_copy, 
                    str(viz_path),
                    scale=2,
                    engine='kaleido',
                    width=800,
                    height=500
                )
                print(f"Debug: Visualization saved successfully to {viz_path}")
                return viz_path
            else:
                print("Debug: No visualization figure provided")
                return None
                
        except Exception as e:
            print(f"Debug: Error saving visualization: {str(e)}")
            import traceback
            print("Debug: Full traceback:")
            print(traceback.format_exc())
            return None
            
    def _clean_temp_files(self):
        """Clean up temporary visualization files"""
        for file in self.temp_dir.glob('visualization_*.png'):
            try:
                file.unlink()
            except Exception:
                pass
                
    def _create_page_template(self, canvas, doc):
        """Draw the page template with Jump branding"""
        width, height = doc.pagesize
        
        # Draw white background
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, width, height, fill=1)
        
        # Draw gradient arc in top-right corner
        p = canvas.beginPath()
        p.moveTo(width-150, height)
        p.curveTo(width-100, height-20, width-50, height-40, width, height-80)
        p.lineTo(width, height)
        p.close()
        canvas.setFillColor(colors.HexColor(JUMP_COLORS['dark_purple']))
        canvas.drawPath(p, fill=1)
        
        # Add Jump logo at bottom left
        logo_path = self.static_dir / 'images' / 'jump_logo_dark.png'
        if logo_path.exists():
            canvas.drawImage(str(logo_path), 72, 30, width=60, height=35, mask='auto')
        
        # Add page number if not first page
        if doc.page > 1:
            canvas.setFont('Helvetica', 9)
            canvas.setFillColor(colors.HexColor(JUMP_COLORS['dark_purple']))
            canvas.drawString(width-40, 30, f"{doc.page}")
            
    def generate_report(self, data, output_path):
        """Generate PDF report using ReportLab"""
        try:
            print("\nDebug: Starting report generation with ReportLab")
            
            # Create the document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72,
                title="Jump's AI Data Analysis Report"
            )
            
            # Story will contain all elements
            story = []
            
            # Title
            story.append(Paragraph("Jump's AI Data Analysis Report", self.styles['JumpTitle']))
            story.append(Spacer(1, 20))
            
            # Query section
            story.append(Paragraph("Query", self.styles['JumpHeading']))
            story.append(Paragraph(data['query_text'], self.styles['JumpBody']))
            story.append(Spacer(1, 20))
            
            # Visualization section
            if 'visualization' in data and data['visualization'] is not None:
                viz_path = self._save_visualization(data['visualization'])
                if viz_path:
                    story.append(Paragraph("Visualization", self.styles['JumpHeading']))
                    img = Image(str(viz_path.absolute()), width=6.5*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 10))
                    
                    if data['viz_suggestion']:
                        story.append(Paragraph(
                            f"Visualization Type: {data['viz_suggestion'].get('type', 'N/A')}",
                            self.styles['JumpBody']
                        ))
                        story.append(Paragraph(
                            data['viz_suggestion'].get('reason', ''),
                            self.styles['JumpBody']
                        ))
                    story.append(Spacer(1, 20))
            
            # Query Results section
            story.append(Paragraph("Query Results", self.styles['JumpHeading']))
            if data['data_columns'] and data['data_rows']:
                table_data = [data['data_columns']] + data['data_rows']
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(JUMP_COLORS['dark_purple'])),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor(JUMP_COLORS['black'])),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(JUMP_COLORS['light_purple'])),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')])
                ])
                table = Table(table_data)
                table.setStyle(table_style)
                story.append(table)
            story.append(Spacer(1, 20))
            
            # Technical Details section
            story.append(Paragraph("Technical Details", self.styles['JumpHeading']))
            story.append(Paragraph("SQL Query:", self.styles['JumpBody']))
            story.append(Paragraph(data['sql_query'], self.styles['JumpCode']))
            story.append(Paragraph(f"Execution Time: {data['execution_time']}", self.styles['JumpBody']))
            story.append(Paragraph(f"Rows Processed: {data['row_count']}", self.styles['JumpBody']))
            
            # Build the document with custom header
            doc.build(story, onFirstPage=self._create_page_template, onLaterPages=self._create_page_template)
            
            print("Debug: Report generated successfully")
            return True
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
            self._clean_temp_files()
            return False
            
    def get_report_as_base64(self, data):
        """Generate report and return as base64 string for Streamlit download"""
        try:
            # Generate report with descriptive filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'Jump_AI_Data_Analysis_Report_{timestamp}.pdf'
            temp_pdf = self.temp_dir / filename
            success = self.generate_report(data, str(temp_pdf))
            
            if not success:
                return None
                
            # Read PDF and convert to base64
            with open(temp_pdf, 'rb') as f:
                pdf_bytes = f.read()
            pdf_b64 = base64.b64encode(pdf_bytes).decode()
            
            # Clean up
            temp_pdf.unlink()
            
            return pdf_b64
            
        except Exception as e:
            print(f"Error generating base64 report: {str(e)}")
            return None
