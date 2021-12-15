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
        public double hand_size = 70;
        public double learn_const = 0.1;
        public int layers_amount = 3;
        public List<List<double>> layers_weights = new List<List<double>>();
        public List<List<double>> neuron = new List<List<double>>();


        public Random randA = new Random();
        public Random randW = new Random();
        public Random randE = new Random();
        public DrawingAttributes attributes = new DrawingAttributes();
        public double alfa, beta;
        public double rand_alfa, rand_beta;
        public MainWindow()
        {
            InitializeComponent();
            attributes.StylusTip = StylusTip.Rectangle;
            attributes.Width = 2;
            attributes.Height = 2;
        }

        public List<double> Weights(List<double> w)
        {
            for (int i = 0; i < 2501; i++)
            {
                w.Add(2.0 * randW.NextDouble() - 1.0);
            }
            return w;
        }
        private new void MouseUp(object sender, MouseButtonEventArgs e)
        {
            double xa, ya, xb, yb;
            alfa = randA.NextDouble() * Math.PI;
            beta = randA.NextDouble() * Math.PI;
            StylusPoint p0, p1, p2;
  
            List<Stroke> l = new List<Stroke>();
            StylusPointCollection points = new StylusPointCollection();

            xa = hand_size * Math.Sin(alfa);
            ya = -hand_size * Math.Cos(alfa) + 130;

            xb = xa + (0 + xa) * Math.Cos(beta) + (ya - 130) * Math.Sin(beta);
            yb = ya + (0 - xa) * Math.Sin(beta) + (ya - 130) * Math.Cos(beta);

            p0 = new StylusPoint(0, 130);
            p1 = new StylusPoint(xa, ya);
            p2 = new StylusPoint(xb, yb);

            points.Add(p0);
            points.Add(p1);
            points.Add(p2);

            Stroke s = new Stroke(points, attributes);
            l.Add(s);
            drawSpace.Strokes.Add(s);
        }

        private new void MouseDown(object sender, MouseButtonEventArgs e)
        {
            drawSpace.Strokes.Clear();
        }

        // funkcja aktywacji
        public double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        // pochodna funkcji aktywacji
        public double SecondSigmoid(double x)
        {
            return x * (1 - x);
        }
        public void RandRadians()
        {
            rand_alfa = randA.NextDouble() * Math.PI;
            rand_beta = randA.NextDouble() * Math.PI;
        }
        public List<double> NormalizeDegrees(List<double> degrees)
        {
            List<double> norm_degrees = new List<double>();
            foreach(double tmp in degrees)
            {
                norm_degrees.Add(0.8 * tmp / Math.PI + 0.1);
            }
            return norm_degrees;
        }
        public List<double> UnnormalizeDegrees(List<double> degrees)
        {
            List<double> unnorm_degrees = new List<double>();
            foreach (double tmp in degrees)
            {
                unnorm_degrees.Add((tmp - 0.1) * Math.PI / 0.8);
            }
            return unnorm_degrees;
        }
    }
}
