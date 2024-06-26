// Complex.cc

/* 
Copyright (C) 1988 Free Software Foundation
    written by Doug Lea (dl@rocky.oswego.edu)

This file is part of the GNU C++ Library.  This library is free
software; you can redistribute it and/or modify it under the terms of
the GNU Library General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your
option) any later version.  This library is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the GNU Library General Public License for more details.
You should have received a copy of the GNU Library General Public
License along with this library; if not, write to the Free Software
Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
*/

// Modified and adapted to QSD software by rschack

//#include "Complex.h"
//#include <stdlib.h>
//
//static const char rcsid[] = "$Id: Complex.cc,v 3.1 1996/11/19 10:05:08 rschack Exp $";
//
//void Complex::error(const char* msg) const
//{
//  cerr << "Fatal Complex arithmetic error. " << msg << endl;
//  exit(1);
//}
//
//double hypotenuse(double a, double b)
////
//// Returns sqrt(a*a+b*b).
//// See Numerical Recipes in C, 2nd edition, p.177, Eq.(5.4.4) and p.949.
//{
//  double x = fabs(a);
//  double y = fabs(b);
//  if( x == 0.0 )
//    return y;
//  else if( y == 0.0 )
//    return x;
//  else if( x > y ) {
//    double tmp = y/x;
//    return x * sqrt( 1.0 + tmp*tmp );
//  }
//  else {
//    double tmp = x/y;
//    return y * sqrt( 1.0 + tmp*tmp );
//  }
//}
//
//Complex& Complex::timesI() {
//  double tmp = im;
//  im = re;
//  re = -tmp;
//  return *this;
//}
//
//Complex& Complex::timesMinusI() {
//  double tmp = im;
//  im = -re;
//  re = tmp;
//  return *this;
//}
//
///* from romine@xagsun.epm.ornl.gov */
//Complex /* const */ operator / (const Complex& x, const Complex& y)
//{
//  double den = fabs(y.real()) + fabs(y.imag());
//  if (den == 0.0) x.error ("Attempted division by zero.");
//  double xrden = x.real() / den;
//  double xiden = x.imag() / den;
//  double yrden = y.real() / den;
//  double yiden = y.imag() / den;
//  double nrm   = yrden * yrden + yiden * yiden;
//  return Complex((xrden * yrden + xiden * yiden) / nrm,
//                 (xiden * yrden - xrden * yiden) / nrm);
//}
//
//Complex& Complex::operator /= (const Complex& y)
//{
//  double den = fabs(y.real()) + fabs(y.imag());
//  if (den == 0.0) error ("Attempted division by zero.");
//  double xrden = re / den;
//  double xiden = im / den;
//  double yrden = y.real() / den;
//  double yiden = y.imag() / den;
//  double nrm   = yrden * yrden + yiden * yiden;
//  re = (xrden * yrden + xiden * yiden) / nrm;
//  im = (xiden * yrden - xrden * yiden) / nrm;
//  return *this;
//}
//
//Complex /* const */ operator / (double x, const Complex& y)
//{
//  double den = norm(y);
//  if (den == 0.0) y.error ("Attempted division by zero.");
//  return Complex((x * y.real()) / den, -(x * y.imag()) / den);
//}
//
//Complex /* const */ operator / (const Complex& x, double y)
//{
//  if (y == 0.0) x.error ("Attempted division by zero.");
//  return Complex(x.real() / y, x.imag() / y);
//}
//
//
//Complex& Complex::operator /= (double y)
//{
//  if (y == 0.0) error ("Attempted division by zero.");
//  re /= y;  im /= y;
//  return *this;
//}
//
//
//Complex /* const */ exp(const Complex& x)
//{
//  double r = exp(x.real());
//  return Complex(r * cos(x.imag()),
//                 r * sin(x.imag()));
//}
//
//Complex /* const */ cosh(const Complex& x)
//{
//  return Complex(cos(x.imag()) * cosh(x.real()),
//                 sin(x.imag()) * sinh(x.real()));
//}
//
//Complex /* const */ sinh(const Complex& x)
//{
//  return Complex(cos(x.imag()) * sinh(x.real()),
//                 sin(x.imag()) * cosh(x.real()));
//}
//
//Complex /* const */ cos(const Complex& x)
//{
//  return Complex(cos(x.real()) * cosh(x.imag()),
//                 -sin(x.real()) * sinh(x.imag()));
//}
//
//Complex /* const */ sin(const Complex& x)
//{
//  return Complex(sin(x.real()) * cosh(x.imag()),
//                 cos(x.real()) * sinh(x.imag()));
//}
//
//Complex /* const */ log(const Complex& x)
//{
//  double h = hypotenuse(x.real(), x.imag());
//  if (h <= 0.0) x.error("attempted log of zero magnitude number.");
//  return Complex(log(h), atan2(x.imag(), x.real()));
//}
//
//// Corrections based on reports from: thc@cs.brown.edu & saito@sdr.slb.com
//Complex /* const */ pow(const Complex& x, const Complex& p)
//{
//  double h = hypotenuse(x.real(), x.imag());
//  if (h <= 0.0) x.error("attempted power of zero magnitude number.");
//
//  double a = atan2(x.imag(), x.real());
//  double lr = pow(h, p.real());
//  double li = p.real() * a;
//  if (p.imag() != 0.0)
//  {
//    lr /= exp(p.imag() * a);
//    li += p.imag() * log(h);
//  }
//  return Complex(lr * cos(li), lr * sin(li));
//}
//
//Complex /* const */ pow(const Complex& x, double p)
//{
//  double h = hypotenuse(x.real(), x.imag());
//  if (h <= 0.0) x.error("attempted power of zero magnitude number.");
//  double lr = pow(h, p);
//  double a = atan2(x.imag(), x.real());
//  double li = p * a;
//  return Complex(lr * cos(li), lr * sin(li));
//}
//
//
//Complex /* const */ sqrt(const Complex& x)
//{
//  if (x.real() == 0.0 && x.imag() == 0.0)
//    return Complex(0.0, 0.0);
//  else
//  {
//    double s = sqrt((fabs(x.real()) + hypotenuse(x.real(), x.imag())) * 0.5);
//    double d = (x.imag() / s) * 0.5;
//    if (x.real() > 0.0)
//      return Complex(s, d);
//    else if (x.imag() >= 0.0)
//      return Complex(d, s);
//    else
//      return Complex(-d, -s);
//  }
//}
//
//
//Complex /* const */ pow(const Complex& x, int p)
//{
//  if (p == 0)
//    return Complex(1.0, 0.0);
//  else if (x == 0.0)
//    return Complex(0.0, 0.0);
//  else
//  {
//    Complex res(1.0, 0.0);
//    Complex b = x;
//    if (p < 0)
//    {
//      p = -p;
//      b = 1.0 / b;
//    }
//    for(;;)
//    {
//      if (p & 1)
//        res *= b;
//      if ((p >>= 1) == 0)
//        return res;
//      else
//        b *= b;
//    }
//  }
//}
//
//ostream& operator << (ostream& s, const Complex& x)
//{
//  return s << "(" << x.real() << ", " << x.imag() << ")" ;
//}
//
//istream& operator >> (istream& s, Complex& x)
//{
//  double r, i;
//  char ch;
//  s >> ws;
//  s.get(ch);
//  if (ch == '(')
//  {
//    s >> r;
//    s >> ws;
//    s.get(ch);
//    if (ch == ',')
//    {
//      s >> i;
//      s >> ws;
//      s.get(ch);
//    }
//    else
//      i = 0;
//    if (ch != ')')
//      s.clear(ios::failbit);
//  }
//  else
//  {
//    s.putback(ch);
//    s >> r;
//    i = 0;
//  }
//  x = Complex(r, i);
//  return s;
//}
