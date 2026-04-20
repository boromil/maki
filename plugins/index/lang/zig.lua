return function(U)
  local get_text = U.get_text
  local find_child = U.find_child
  local compact_ws = U.compact_ws
  local new_entry = U.new_entry
  local new_import_entry = U.new_import_entry
  local SECTION = U.SECTION
  local CHILD_BRIEF = U.CHILD_BRIEF
  local FIELD_TRUNCATE_THRESHOLD = U.FIELD_TRUNCATE_THRESHOLD
  local truncated_msg = U.truncated_msg

  local function split_path(s)
    local parts = {}
    for p in s:gmatch("[^/]+") do
      parts[#parts + 1] = p
    end
    return parts
  end

  local function get_qualifiers(node, source)
    local parts = {}
    for _, child in ipairs(node:children()) do
      local ck = child:type()
      if ck == "pub" or ck == "export" or ck == "extern"
        or ck == "inline" or ck == "noinline" then
        parts[#parts + 1] = get_text(child, source)
      end
    end
    return table.concat(parts, " ")
  end

  local function fn_sig(node, source)
    local name_node = node:field("name")[1]
    if not name_node then return nil end
    local name = get_text(name_node, source)
    local params = find_child(node, "parameters")
    local params_text = params and get_text(params, source) or "()"
    local ret_node = node:field("type")[1]
    local ret = ret_node and (" " .. get_text(ret_node, source)) or ""
    return compact_ws(name .. params_text .. ret)
  end

  local function find_import_call(node, source)
    for _, child in ipairs(node:children()) do
      if child:type() == "builtin_function" then
        local bi = find_child(child, "builtin_identifier")
        if bi and get_text(bi, source) == "@import" then
          return child
        end
      end
    end
    return nil
  end

  local function find_container(node)
    for _, child in ipairs(node:children()) do
      local ck = child:type()
      if ck == "struct_declaration" or ck == "enum_declaration"
        or ck == "union_declaration" or ck == "opaque_declaration" then
        return child, ck
      end
    end
    return nil, nil
  end

  local function extract_struct_fields(container, source)
    local fields = {}
    local total = 0
    for _, child in ipairs(container:children()) do
      if child:type() == "container_field" then
        total = total + 1
        if total <= FIELD_TRUNCATE_THRESHOLD then
          local name_node = child:field("name")[1]
          local type_node = child:field("type")[1]
          local fname = name_node and get_text(name_node, source) or "_"
          local ftype = type_node and (" " .. get_text(type_node, source)) or ""
          fields[#fields + 1] = compact_ws(fname .. ":" .. ftype)
        end
      end
    end
    if total > FIELD_TRUNCATE_THRESHOLD and #fields < total then
      fields[#fields + 1] = truncated_msg(total)
    end
    return fields
  end

  local function extract_enum_variants(container, source)
    local fields = {}
    local total = 0
    for _, child in ipairs(container:children()) do
      if child:type() == "container_field" then
        total = total + 1
        if total <= FIELD_TRUNCATE_THRESHOLD then
          local name_node = child:field("name")[1]
          fields[#fields + 1] = name_node and get_text(name_node, source) or "_"
        end
      end
    end
    if total > FIELD_TRUNCATE_THRESHOLD and #fields < total then
      fields[#fields + 1] = truncated_msg(total)
    end
    return fields
  end

  local function var_name(node, source)
    local id = find_child(node, "identifier")
    return id and get_text(id, source) or nil
  end

  return {
    import_separator = "/",

    is_doc_comment = function(node, source)
      if node:type() ~= "comment" then return false end
      local text = get_text(node, source)
      return text:sub(1, 3) == "///" and text:sub(1, 4) ~= "////"
    end,

    is_module_doc = function(node, source)
      if node:type() ~= "comment" then return false end
      local text = get_text(node, source)
      return text:sub(1, 3) == "//!"
    end,

    is_test_node = function(node, _source, _attrs)
      return node:type() == "test_declaration"
    end,

    extract_nodes = function(node, source, _attrs)
      local kind = node:type()

      if kind == "function_declaration" then
        local sig = fn_sig(node, source)
        if not sig then return {} end
        local q = get_qualifiers(node, source)
        local text = q ~= "" and (q .. " fn " .. sig) or ("fn " .. sig)
        return { new_entry(SECTION.Function, node, text) }

      elseif kind == "variable_declaration" then
        local name = var_name(node, source)
        if not name then return {} end

        local q = get_qualifiers(node, source)
        local q_prefix = q ~= "" and (q .. " ") or ""

        local is_var = false
        for _, child in ipairs(node:children()) do
          if child:type() == "var" then
            is_var = true
            break
          end
        end
        local kw = is_var and "var " or "const "

        local import_call = find_import_call(node, source)
        if import_call then
          local args = find_child(import_call, "arguments")
          if args then
            local str_node = find_child(args, "string")
            if str_node then
              local raw = get_text(str_node, source)
              local path = raw:match('^"(.-)"$') or raw
              path = path:gsub("%.zig$", "")
              return { new_import_entry(node, { split_path(path) }) }
            end
          end
          return {}
        end

        local container, ckind = find_container(node)
        if container then
          local type_kw = ckind:gsub("_declaration$", "")
          local entry = new_entry(SECTION.Type, node, q_prefix .. type_kw .. " " .. name)
          if ckind == "enum_declaration" then
            entry.children = extract_enum_variants(container, source)
            entry.child_kind = CHILD_BRIEF
          else
            entry.children = extract_struct_fields(container, source)
          end
          return { entry }
        end

        local type_node = node:field("type")[1]
        local type_str = type_node and (": " .. get_text(type_node, source)) or ""
        local text = compact_ws(q_prefix .. kw .. name .. type_str)
        return { new_entry(SECTION.Constant, node, text) }
      end

      return {}
    end,
  }
end
