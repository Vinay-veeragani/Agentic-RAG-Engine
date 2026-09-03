"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function CollectionSelect({
  value,
  onChange,
  placeholder = "All collections",
}: {
  value: string | null;
  onChange: (id: string | null) => void;
  placeholder?: string;
}) {
  const { data: collections } = useQuery({
    queryKey: ["collections"],
    queryFn: api.collections.list,
  });

  return (
    <Select
      value={value ?? "__all__"}
      onValueChange={(v) => onChange(v === "__all__" ? null : v)}
    >
      <SelectTrigger className="w-64">
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="__all__">{placeholder}</SelectItem>
        {collections?.map((c) => (
          <SelectItem key={c.id} value={c.id}>
            {c.name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
