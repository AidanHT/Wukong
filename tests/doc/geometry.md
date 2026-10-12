# Module `tests.doc.geometry`

Small geometry helpers, documented and tested by their own examples.

```wk
assert(area_of(Rect { w: 2, h: 3 }) == 6);
```

## `struct Rect`

```wk
struct Rect {
    w: i32,
    h: i32,
}
```

A rectangle with integer sides.

### Fields

- `w: i32`

  The width.

- `h: i32`

  The height.

## `fn area_of`

```wk
fn area_of(r: Rect) -> i32
```

The area of `r`.

```wk
let r = Rect { w: 4, h: 5 };
assert(area_of(r) == 20);
print(area_of(r));
```

## `fn double`

```wk
fn double(x: i32) -> i32
```

Doubles `x`, which must be an `i32`:

```wk,compile_fail,E0401
let y = double(1.5);
```

An example with its own `main`:

```wk
fn main() -> i32 {
    if double(21) != 42 {
        return 1;
    }
    return 0;
}
```

## `enum Facing`

```wk
enum Facing {
    Front,
    Back,
}
```

Which way a shape faces.

### Variants

- `Front`

  Toward the viewer.

- `Back`

## `fn spin`

```wk
fn spin()
```

A loop that never ends: compiled, never run.

```wk,no_run
spin();
```

A sketch of what is not Wukong yet:

```wk,ignore
spin(forever = true);
```

```text
not an example at all
```

## `fn main`

```wk
fn main() -> i32
```

## `fn spin_if`

```wk
fn spin_if(go: bool)
```

Spins when `go` is true.
