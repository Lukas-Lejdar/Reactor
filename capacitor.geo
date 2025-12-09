r0 = 0.5;
r2 = 1.0;
theta = Pi/6;
lc = 0.3;

c = {0,0,0};

p0_s = {r0,0,0};
p1_s = {r0*Cos(theta), r0*Sin(theta),0};

p0_l = {r2,0,0};
p1_l = {r2*Cos(theta), r2*Sin(theta),0};

Point(1) = {p0_s[0],p0_s[1],0, lc};
Point(2) = {p1_s[0],p1_s[1],0, lc};
Point(3) = {p0_l[0],p0_l[1],0, lc};
Point(4) = {p1_l[0],p1_l[1],0, lc};
Point(5) = {0,0,0, lc};

Circle(1) = {1,5,2};
Circle(2) = {3,5,4};

Line(3) = {1,3};
Line(4) = {2,4};

Line Loop(1) = {1,4,-2,-3};

// Surface
Plane Surface(1) = {1};
Recombine Surface {1};


Physical Curve("inner_circle") = {1};
Physical Curve("outer_circle") = {2};
Physical Curve("outer_lines") = {3,4};

Physical Surface("domain") = {1};
