// Platonisches Dodekaeder – mit Kugeln an jedem Stabende

$fn = 60;
edge = 80;
rod_d = 8;
phi = (1 + sqrt(5)) / 2;

pts_raw = [
    [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
    [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
    [ phi,  1/phi, 0], [ phi, -1/phi, 0], [-phi,  1/phi, 0], [-phi, -1/phi, 0],
    [0,  phi,  1/phi], [0,  phi, -1/phi], [0, -phi,  1/phi], [0, -phi, -1/phi],
    [ 1/phi, 0,  phi], [ 1/phi, 0, -phi], [-1/phi, 0,  phi], [-1/phi, 0, -phi]
];

raw_edge = norm(pts_raw[0] - pts_raw[12]);
sc = edge / raw_edge;
v = [for(p = pts_raw) p * sc];

faces = [
    [0, 12, 4, 18, 16],
    [0, 12, 13, 1, 8],
    [0, 8, 9, 2, 16],
    [2, 14, 6, 18, 16],
    [12, 4, 10, 5, 13],
    [11, 10, 5, 19, 7],
    [13, 5, 19, 17, 1],
    [1, 8, 9, 3, 17],
    [11, 6, 18, 4, 10],
    [15, 14, 6, 11, 7],
    [15, 3, 17, 19, 7],
    [14, 2, 9, 3, 15]
];

module rod(a, b) {
    vec = b - a;
    len = norm(vec);
    union() {
        translate(a)
            rotate([0, acos(vec[2]/len), atan2(vec[1], vec[0])])
                cylinder(d = rod_d, h = len);
        translate(a) sphere(d = rod_d * 1.2);
        translate(b) sphere(d = rod_d * 1.2);
    }
}

union() {
    for(face = faces) {
        verts = [for(i = face) v[i]];
        for(i = [0:4]) {
            rod(verts[i], verts[(i+1)%5]);
        }
    }
}