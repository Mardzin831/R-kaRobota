using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace RękaRobota
{
    public partial class MainWindow : Window
    {
        public DrawingAttributes attributes = new DrawingAttributes();
        public MainWindow()
        {
            InitializeComponent();
            attributes.StylusTip = StylusTip.Rectangle;
            attributes.Width = 2;
            attributes.Height = 2;

            //StylusPoint p, q;
            //List<Stroke> l = new List<Stroke>();
            //StylusPointCollection points = new StylusPointCollection();
                
            //p = new StylusPoint(100, 100);
            //q = new StylusPoint(0, 120);
            //points.Add(p);
            //points.Add(q);
            //Stroke s = new Stroke(points, attributes);
            //l.Add(s);
            //drawSpace.Strokes.Add(s);
        }

        private new void MouseUp(object sender, MouseButtonEventArgs e)
        {
            StylusPoint p, q;
            var tmp = drawSpace.Strokes.Last();
            List<Stroke> l = new List<Stroke>();
            StylusPointCollection points = new StylusPointCollection();

            p = tmp.StylusPoints.Last();
            q = new StylusPoint(0, 120);
            points.Add(p);
            points.Add(q);
            Stroke s = new Stroke(points, attributes);
            l.Add(s);

            drawSpace.Strokes.Add(s);
        }

        private new void MouseDown(object sender, MouseButtonEventArgs e)
        {
            drawSpace.Strokes.Clear();
        }
    }
}
